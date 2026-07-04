#!/usr/bin/env python3
"""
LLM extraction accuracy benchmark.

Runs LLMProspectusAnalyzer's core extraction methods (_extract_financial_metrics,
_perform_benchmarking_analysis, _analyze_ipo_specifics) against hand-verified
fixtures under fixtures/<company>/ and scores the output against expected.json,
producing a JSON + markdown report under reports/.

This bypasses the UI entirely and makes real LLM API calls - re-run it whenever
prompts/models change, not as part of normal test runs.

Usage:
    python benchmarks/extraction_accuracy/run.py
    python benchmarks/extraction_accuracy/run.py --providers groq,openai
    python benchmarks/extraction_accuracy/run.py --fixtures vidya_wires
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from loguru import logger
from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer

BENCHMARK_DIR = Path(__file__).parent
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
REPORTS_DIR = BENCHMARK_DIR / "reports"

# Provider -> env var required to actually run it
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def to_plain(obj):
    """Recursively convert dataclasses to plain dicts for JSON serialization / substring search."""
    if is_dataclass(obj):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def normalize_pct(value):
    """A model might return a ratio as 0.0216 or as 2.16 - normalize to percentage scale."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value * 100 if abs(value) <= 1.5 else value


def check_numeric_tolerance(check_name, actual, expected, tolerance_pct, why):
    if expected is None:
        return None
    if actual is None:
        return {"check": check_name, "result": "FAIL", "why": why,
                 "detail": f"expected ~{expected}, got None (not extracted)"}
    actual_norm = normalize_pct(actual)
    if actual_norm is None:
        return {"check": check_name, "result": "FAIL", "why": why,
                 "detail": f"expected ~{expected}, got non-numeric value {actual!r} "
                           "(likely an unevaluated arithmetic expression string)"}
    diff = abs(actual_norm - expected)
    passed = diff <= tolerance_pct
    return {"check": check_name, "result": "PASS" if passed else "FAIL", "why": why,
             "detail": f"expected {expected} (tolerance +/-{tolerance_pct}), got raw={actual} normalized={actual_norm:.2f}, diff={diff:.2f}"}


def check_should_be_null(check_name, actual, why):
    passed = actual is None
    return {"check": check_name, "result": "PASS" if passed else "FAIL", "why": why,
             "detail": f"expected None (not disclosed/computable), got {actual!r}"}


def check_directional(check_name, actual, expected_positive, why):
    if actual is None:
        return {"check": check_name, "result": "FAIL", "why": why,
                 "detail": "expected a positive-growth value, got None (not extracted)"}
    actual_norm = normalize_pct(actual)
    if actual_norm is None:
        return {"check": check_name, "result": "FAIL", "why": why,
                 "detail": f"got non-numeric value {actual!r} - not evaluated to a number "
                           "(likely an unevaluated arithmetic expression string)"}
    passed = (actual_norm > 0) == expected_positive
    return {"check": check_name, "result": "PASS" if passed else "FAIL", "why": why,
             "detail": f"got raw={actual} normalized={actual_norm:.2f}"}


def check_fact_recall(check_name, haystack_text, needles, why):
    haystack_lower = haystack_text.lower()
    found = [n for n in needles if n.lower() in haystack_lower]
    passed = len(found) > 0
    return {"check": check_name, "result": "PASS" if passed else "FAIL", "why": why,
             "detail": f"looked for any of {needles}, found {found}"}


def check_price_band_not_fabricated(ipo_specifics_plain, why):
    pricing = ipo_specifics_plain.get("ipo_pricing_analysis", {}) if isinstance(ipo_specifics_plain, dict) else {}
    price_band_value = str(pricing.get("price_band", "")) if isinstance(pricing, dict) else ""
    fabricated_number = bool(re.search(r"\d", price_band_value)) and "not disclosed" not in price_band_value.lower()
    return {"check": "price_band_not_fabricated", "result": "FAIL" if fabricated_number else "PASS", "why": why,
             "detail": f"price_band field returned: {price_band_value!r}"}


def score_financial_metrics(metrics, expected):
    fm = expected["financial_metrics"]
    tol = expected["tolerance_pct"]
    checks = [
        check_numeric_tolerance(
            "net_profit_margin_fy2024", metrics.net_profit_margin,
            fm["net_profit_margin_fy2024_pct"], tol * 3,
            "PAT/Total income for FY2024, hand-computed from restated financials as 256.93/11884.89 = 2.16%. "
            "Wide tolerance since the model may pick a slightly different period."
        ),
        check_directional(
            "revenue_growth_3yr_direction", metrics.revenue_growth_3yr, True,
            f"Real revenue grew {fm['revenue_growth_fy23_to_fy24_pct']}% YoY (FY23->FY24) and "
            f"{fm['revenue_growth_fy22_to_fy23_pct']}% (FY22->FY23) per the restated financials - "
            "growth was clearly positive across all periods, so any extracted value should be positive too."
        ),
        check_directional(
            "profit_growth_3yr_direction", metrics.profit_growth_3yr, True,
            f"Real PAT grew {fm['profit_growth_fy23_to_fy24_pct']}% YoY (FY23->FY24) per the restated financials."
        ),
        check_should_be_null(
            "trailing_pe_ratio_abstains", metrics.trailing_pe_ratio,
            "This is a Draft Red Herring Prospectus - price band is explicitly not yet set "
            "(\"THE PRICE BAND ... WILL BE DECIDED\"), so P/E is not computable. A correct extractor "
            "should return null rather than fabricate a number."
        ),
        check_should_be_null(
            "price_to_book_ratio_abstains", metrics.price_to_book_ratio,
            "Price band not disclosed - P/B is not computable pre-pricing."
        ),
        check_should_be_null(
            "ev_to_ebitda_ratio_abstains", metrics.ev_to_ebitda_ratio,
            "Price band not disclosed - EV/EBITDA needs market cap, not computable pre-pricing."
        ),
    ]
    return [c for c in checks if c is not None]


def score_ipo_specifics(ipo_specifics, expected):
    plain = to_plain(ipo_specifics)
    serialized = json.dumps(plain, default=str)
    ios = expected["ipo_specifics"]

    checks = [
        check_price_band_not_fabricated(
            plain,
            "Draft filing - price band genuinely not yet set in the source document."
        ),
        check_fact_recall(
            "use_of_funds_mentions_alcu_capex", serialized, ["1,400", "1400"],
            f"Real use-of-funds item: {ios['use_of_funds_inr_million'][0]['purpose']} "
            f"= Rs {ios['use_of_funds_inr_million'][0]['amount']} million, stated explicitly in the prospectus."
        ),
        check_fact_recall(
            "use_of_funds_mentions_debt_repayment", serialized, ["1,000", "1000"],
            f"Real use-of-funds item: {ios['use_of_funds_inr_million'][1]['purpose']} "
            f"= Rs {ios['use_of_funds_inr_million'][1]['amount']} million, stated explicitly in the prospectus."
        ),
        check_fact_recall(
            "lead_managers_pantomath", serialized, ["pantomath"],
            f"Real lead manager per the prospectus cover page: {ios['lead_managers'][0]}."
        ),
        check_fact_recall(
            "lead_managers_idbi", serialized, ["idbi"],
            f"Real lead manager per the prospectus cover page: {ios['lead_managers'][1]}."
        ),
    ]
    return checks


def run_provider_on_fixture(provider, fixture_name, prospectus_text, expected):
    logger.info(f"[{provider}] Running extraction on fixture '{fixture_name}'")
    company_name = expected["company_name"]
    sector = expected.get("sector", "Unknown")

    tmp_db_path = tempfile.mkdtemp(prefix=f"ipo_benchmark_vecdb_{provider}_")
    result = {
        "provider": provider,
        "fixture": fixture_name,
        "checks": [],
        "errors": [],
        "timings_sec": {},
    }
    try:
        t0 = time.time()
        analyzer = LLMProspectusAnalyzer(provider=provider, use_vector_db=True, db_path=tmp_db_path)
        if analyzer.client is None:
            result["errors"].append(f"{provider} client failed to initialize (missing/invalid API key)")
            return result

        analyzer.chunk_and_store_prospectus(prospectus_text, company_name, sector)
        result["timings_sec"]["chunk_and_store"] = round(time.time() - t0, 1)

        t1 = time.time()
        try:
            metrics = analyzer._extract_financial_metrics(prospectus_text, company_name)
            result["checks"].extend(score_financial_metrics(metrics, expected))
            result["raw_financial_metrics"] = to_plain(metrics)
        except Exception as e:
            result["errors"].append(f"_extract_financial_metrics failed: {e}")
        result["timings_sec"]["financial_metrics"] = round(time.time() - t1, 1)

        t2 = time.time()
        try:
            ipo_specifics = analyzer._analyze_ipo_specifics(prospectus_text, company_name)
            result["checks"].extend(score_ipo_specifics(ipo_specifics, expected))
            result["raw_ipo_specifics"] = to_plain(ipo_specifics)
        except Exception as e:
            result["errors"].append(f"_analyze_ipo_specifics failed: {e}")
        result["timings_sec"]["ipo_specifics"] = round(time.time() - t2, 1)

        t3 = time.time()
        try:
            benchmarking = analyzer._perform_benchmarking_analysis(prospectus_text, company_name, sector)
            result["raw_benchmarking"] = to_plain(benchmarking)
            # No hand-verified ground truth for competitive positioning (subjective) -
            # recorded for manual review only, not scored pass/fail.
        except Exception as e:
            result["errors"].append(f"_perform_benchmarking_analysis failed: {e}")
        result["timings_sec"]["benchmarking"] = round(time.time() - t3, 1)

        result["timings_sec"]["total"] = round(time.time() - t0, 1)

    finally:
        shutil.rmtree(tmp_db_path, ignore_errors=True)

    return result


def available_providers():
    return [p for p, env_key in PROVIDER_ENV_KEYS.items()
            if os.getenv(env_key) and not os.getenv(env_key, "").startswith("your_")]


def build_markdown_report(all_results, generated_at):
    lines = [f"# LLM Extraction Accuracy Report", "", f"Generated: {generated_at}", ""]
    for res in all_results:
        header = f"## {res['provider']} / {res['fixture']}"
        lines.append(header)
        lines.append("")
        if res["errors"]:
            lines.append("**Errors:**")
            for e in res["errors"]:
                lines.append(f"- {e}")
            lines.append("")
        checks = res["checks"]
        n_pass = sum(1 for c in checks if c["result"] == "PASS")
        n_total = len(checks)
        lines.append(f"**Score: {n_pass}/{n_total} checks passed**")
        lines.append("")
        lines.append("| Check | Result | Detail |")
        lines.append("|---|---|---|")
        for c in checks:
            icon = "PASS" if c["result"] == "PASS" else "FAIL"
            lines.append(f"| {c['check']} | {icon} | {c['detail']} |")
        lines.append("")
        timings = res.get("timings_sec", {})
        if timings:
            lines.append(f"Timing: {timings}")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LLM extraction accuracy benchmark")
    parser.add_argument("--providers", default=None,
                         help="Comma-separated providers to test (default: auto-detect from configured API keys)")
    parser.add_argument("--fixtures", default=None,
                         help="Comma-separated fixture names (default: all fixtures under fixtures/)")
    args = parser.parse_args()

    providers = args.providers.split(",") if args.providers else available_providers()
    if not providers:
        print("No providers with configured API keys found. Set at least one of "
              f"{list(PROVIDER_ENV_KEYS.values())} in .env")
        sys.exit(1)

    fixture_names = (args.fixtures.split(",") if args.fixtures
                      else [d.name for d in FIXTURES_DIR.iterdir() if d.is_dir()])

    print(f"Providers: {providers}")
    print(f"Fixtures: {fixture_names}")

    all_results = []
    for fixture_name in fixture_names:
        fdir = FIXTURES_DIR / fixture_name
        prospectus_text = (fdir / "prospectus.txt").read_text(encoding="utf-8")
        expected = json.loads((fdir / "expected.json").read_text(encoding="utf-8"))

        for provider in providers:
            res = run_provider_on_fixture(provider, fixture_name, prospectus_text, expected)
            all_results.append(res)
            n_pass = sum(1 for c in res["checks"] if c["result"] == "PASS")
            n_total = len(res["checks"])
            print(f"  [{provider}/{fixture_name}] {n_pass}/{n_total} checks passed"
                  f"{' - ERRORS: ' + '; '.join(res['errors']) if res['errors'] else ''}")

    generated_at = datetime.now().isoformat()
    run_dir = REPORTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(build_markdown_report(all_results, generated_at), encoding="utf-8")

    print(f"\nReport written to {run_dir}/report.md and {run_dir}/results.json")


if __name__ == "__main__":
    main()
