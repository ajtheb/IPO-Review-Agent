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
from types import SimpleNamespace

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import copy

from loguru import logger
from src.agent import IPOReviewAgent
from src.analyzers import RiskAnalyzer, BusinessAnalyzer
from src.analyzers.llm_prospectus_analyzer import LLMProspectusAnalyzer, MAX_REFLECTION_RETRIES
from src.models import NewsAnalysis

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


def _latest_margin(fm):
    """Find the (year, key, value) for the most recent net_profit_margin_fy<YYYY>_pct key.
    Doesn't assume a specific fiscal year, so this works for fixtures whose latest reported
    year isn't FY2024 (e.g. a fixture built from an FY2025 restated financial summary)."""
    pattern = re.compile(r"^net_profit_margin_fy(\d{4})_pct$")
    candidates = [(int(m.group(1)), k, fm[k]) for k in fm if (m := pattern.match(k))]
    if not candidates:
        return None, None, None
    return max(candidates, key=lambda c: c[0])


def _growth_windows_desc(fm, metric_prefix):
    """Human-readable description of every '<metric_prefix>_fyXX_to_fyYY_pct' key present, for the check's why-text."""
    pattern = re.compile(rf"^{metric_prefix}_fy(\d{{2,4}})_to_fy(\d{{2,4}})_pct$")
    windows = sorted(
        ((int(m.group(2)), int(m.group(1)), fm[k]) for k in fm if (m := pattern.match(k))),
        reverse=True,
    )
    return " and ".join(f"{v}% (FY{s}->FY{e})" for e, s, v in windows)


def _latest_growth(fm, metric_prefix):
    """Find the (end_year, start_year, value) for the most recent '<prefix>_fyXX_to_fyYY_pct' key.
    Fixture years are written as 2-digit ('fy23_to_fy24'), unlike the 4-digit margin keys, so this
    intentionally accepts both widths."""
    pattern = re.compile(rf"^{metric_prefix}_fy(\d{{2,4}})_to_fy(\d{{2,4}})_pct$")
    candidates = [(int(m.group(2)), int(m.group(1)), fm[k]) for k in fm if (m := pattern.match(k))]
    if not candidates:
        return None, None, None
    return max(candidates, key=lambda c: c[0])


def score_financial_metrics(metrics, expected):
    fm = expected["financial_metrics"]
    tol = expected["tolerance_pct"]
    checks = []

    margin_year, margin_key, margin_value = _latest_margin(fm)
    if margin_key:
        checks.append(check_numeric_tolerance(
            f"net_profit_margin_fy{margin_year}", metrics.net_profit_margin, margin_value, tol * 3,
            f"PAT/income for FY{margin_year}, hand-verified from the restated financials = {margin_value}%. "
            "Wide tolerance since the model may pick a slightly different period or income base "
            "(Total income vs. Revenue from Operations)."
        ))

    revenue_positive = fm.get("revenue_growth_3yr_direction_positive", True)
    revenue_desc = _growth_windows_desc(fm, "revenue_growth")
    checks.append(check_directional(
        "revenue_growth_3yr_direction", metrics.revenue_growth_3yr, revenue_positive,
        (f"Real revenue grew {revenue_desc} per the restated financials - "
         f"growth was clearly {'positive' if revenue_positive else 'negative overall'} across the reported "
         "period, so any extracted value should match that sign.") if revenue_desc else
        "Directional sign hand-verified from the restated financials."
    ))

    profit_positive = fm.get("profit_growth_3yr_direction_positive", True)
    profit_desc = _growth_windows_desc(fm, "profit_growth")
    checks.append(check_directional(
        "profit_growth_3yr_direction", metrics.profit_growth_3yr, profit_positive,
        (f"Real PAT changed {profit_desc} per the restated financials - the overall multi-year trend is "
         f"{'positive' if profit_positive else 'negative'} even though not every individual year moved the same way.") if profit_desc else
        "Directional sign hand-verified from the restated financials."
    ))

    checks.append(check_should_be_null(
        "trailing_pe_ratio_abstains", metrics.trailing_pe_ratio,
        "This is a Draft Red Herring Prospectus - price band is explicitly not yet set "
        "(\"THE PRICE BAND ... WILL BE DECIDED\"), so P/E is not computable. A correct extractor "
        "should return null rather than fabricate a number."
    ))
    checks.append(check_should_be_null(
        "price_to_book_ratio_abstains", metrics.price_to_book_ratio,
        "Price band not disclosed - P/B is not computable pre-pricing."
    ))
    checks.append(check_should_be_null(
        "ev_to_ebitda_ratio_abstains", metrics.ev_to_ebitda_ratio,
        "Price band not disclosed - EV/EBITDA needs market cap, not computable pre-pricing."
    ))

    return [c for c in checks if c is not None]


# Shared, stateless instances - RiskAnalyzer/BusinessAnalyzer have no __init__ side effects,
# unlike IPOReviewAgent (which would spin up DataSourceManager/LLM clients), so these are
# cheap to keep at module scope rather than rebuilding per fixture/provider.
_RISK_ANALYZER = RiskAnalyzer()
_BUSINESS_ANALYZER = BusinessAnalyzer()
_NEUTRAL_NEWS = NewsAnalysis(sentiment_score=0.0, key_themes=[])


def compute_verdict(profit_margin_pct, revenue_growth_pct, sector):
    """Run a (profit margin %, revenue growth %) pair through the same production
    Strong Buy/Buy/Hold/Avoid formula as IPOReviewAgent.analyze_ipo (src/agent.py:
    _predict_listing_gains / _calculate_long_term_score / _generate_recommendation).

    Those three methods don't touch `self`, so they're called unbound (self=None) to avoid
    constructing a real IPOReviewAgent, which would spin up DataSourceManager/LLM clients.
    News sentiment is held neutral (no live scraping here) and market_cap uses a fixed
    mid-range placeholder, since price band is undisclosed pre-IPO in every fixture - both
    fixtures/providers see the same neutral inputs, so any verdict difference is attributable
    to the financial-metric extraction being benchmarked, not to these placeholders.
    """
    metrics = SimpleNamespace(
        profit_margin=None if profit_margin_pct is None else profit_margin_pct / 100,
        revenue_growth_rate=None if revenue_growth_pct is None else revenue_growth_pct / 100,
    )
    company_info = {"sector": sector, "market_cap": 1_000_000_000}

    risk_assessment = _RISK_ANALYZER.assess_risks(metrics, {}, _NEUTRAL_NEWS, company_info)
    strengths_weaknesses = _BUSINESS_ANALYZER.analyze_business_fundamentals(company_info, metrics, {})
    listing_gain = IPOReviewAgent._predict_listing_gains(None, metrics, _NEUTRAL_NEWS, risk_assessment)
    long_term_score = IPOReviewAgent._calculate_long_term_score(None, metrics, risk_assessment, strengths_weaknesses)
    recommendation = IPOReviewAgent._generate_recommendation(None, listing_gain, long_term_score, risk_assessment)

    return {
        "recommendation": recommendation.value,
        "long_term_score": round(long_term_score, 2),
        "listing_gain_prediction_pct": round(listing_gain, 2),
        "overall_risk": risk_assessment.overall_risk.value,
    }


def score_final_verdict(metrics, expected):
    """Feed both the hand-verified ground-truth financials and the model's extracted
    financials through compute_verdict(), and check whether an extraction slip on profit
    margin / revenue growth would flip the final Strong Buy/Buy/Hold/Avoid call an investor
    would actually see - a user-facing failure even when the underlying numbers are only
    off by a little."""
    fm = expected["financial_metrics"]
    sector = expected.get("sector", "Unknown")

    _, _, margin_value = _latest_margin(fm)
    _, _, growth_value = _latest_growth(fm, "revenue_growth")
    truth_verdict = compute_verdict(margin_value, growth_value, sector)

    extracted_margin = normalize_pct(metrics.net_profit_margin)
    extracted_growth = normalize_pct(metrics.revenue_growth_3yr)
    extracted_verdict = compute_verdict(extracted_margin, extracted_growth, sector)

    passed = extracted_verdict["recommendation"] == truth_verdict["recommendation"]
    check = {
        "check": "final_verdict_matches_ground_truth",
        "result": "PASS" if passed else "FAIL",
        "why": ("Runs both the hand-verified financials and the model's extracted financials "
                "through the production recommendation formula (src/agent.py _generate_recommendation). "
                f"Ground truth: margin={margin_value}%, revenue_growth={growth_value}% -> "
                f"{truth_verdict['recommendation']} (long_term_score={truth_verdict['long_term_score']}/10)."),
        "detail": (f"extracted: margin={extracted_margin}%, revenue_growth={extracted_growth}% -> "
                   f"{extracted_verdict['recommendation']} (long_term_score={extracted_verdict['long_term_score']}/10) "
                   f"vs. ground truth "
                   f"{truth_verdict['recommendation']}"),
    }
    return check, extracted_verdict, truth_verdict


def _needles_for_amount(amount):
    """Substring forms (comma and non-comma) a model's raw JSON output might use for a disclosed amount."""
    if amount is None:
        return None
    if isinstance(amount, float) and amount.is_integer():
        amount = int(amount)
    if isinstance(amount, int):
        return sorted({str(amount), f"{amount:,}"})
    return sorted({str(amount)})


def score_ipo_specifics(ipo_specifics, expected):
    """Data-driven scoring: every check is derived from expected.json's ipo_specifics block,
    so this works unmodified for any fixture rather than one hardcoded company's facts."""
    plain = to_plain(ipo_specifics)
    serialized = json.dumps(plain, default=str)
    ios = expected["ipo_specifics"]

    checks = [
        check_price_band_not_fabricated(
            plain,
            "Draft filing - price band genuinely not yet set in the source document."
        ),
    ]

    use_of_funds_key = next((k for k in ios if k.startswith("use_of_funds") and isinstance(ios[k], list)), None)
    if use_of_funds_key:
        unit = use_of_funds_key.replace("use_of_funds_", "").replace("_", " ") or "units"
        for item in ios[use_of_funds_key]:
            needles = _needles_for_amount(item.get("amount"))
            if not needles:
                continue
            slug = re.sub(r"[^a-z0-9]+", "_", item["purpose"].lower()).strip("_")[:40]
            checks.append(check_fact_recall(
                f"use_of_funds_mentions_{slug}", serialized, needles,
                f"Real use-of-funds item: {item['purpose']} "
                f"= Rs {item['amount']} {unit}, stated explicitly in the prospectus."
            ))

    for lm in ios.get("lead_managers", []):
        token = lm.split()[0].lower()
        slug = re.sub(r"[^a-z0-9]+", "_", lm.lower()).strip("_")[:30]
        checks.append(check_fact_recall(
            f"lead_manager_{slug}", serialized, [token],
            f"Real lead manager per the prospectus cover page: {lm}."
        ))

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
        metrics = None
        try:
            metrics = analyzer._extract_financial_metrics(prospectus_text, company_name)
            baseline_metrics = copy.deepcopy(metrics)
            result["checks_before_reflection"] = score_financial_metrics(baseline_metrics, expected)
            result["raw_financial_metrics_before_reflection"] = to_plain(baseline_metrics)

            try:
                verdict_check, extracted_verdict, truth_verdict = score_final_verdict(baseline_metrics, expected)
                result["final_verdict_before_reflection"] = {
                    "extracted": extracted_verdict, "ground_truth": truth_verdict
                }
            except Exception as e:
                result["errors"].append(f"score_final_verdict (before reflection) failed: {e}")
        except Exception as e:
            result["errors"].append(f"_extract_financial_metrics failed: {e}")
        result["timings_sec"]["financial_metrics"] = round(time.time() - t1, 1)

        t2 = time.time()
        try:
            ipo_specifics = analyzer._analyze_ipo_specifics(prospectus_text, company_name)
            result["checks"].extend(score_ipo_specifics(ipo_specifics, expected))
            result["raw_ipo_specifics"] = to_plain(ipo_specifics)
        except Exception as e:
            ipo_specifics = None
            result["errors"].append(f"_analyze_ipo_specifics failed: {e}")
        result["timings_sec"]["ipo_specifics"] = round(time.time() - t2, 1)

        t3 = time.time()
        try:
            benchmarking = analyzer._perform_benchmarking_analysis(prospectus_text, company_name, sector)
            result["raw_benchmarking"] = to_plain(benchmarking)
            # No hand-verified ground truth for competitive positioning (subjective) -
            # recorded for manual review only, not scored pass/fail.
        except Exception as e:
            benchmarking = None
            result["errors"].append(f"_perform_benchmarking_analysis failed: {e}")
        result["timings_sec"]["benchmarking"] = round(time.time() - t3, 1)

        # STEP 4/4 equivalent: exercise the same self-critique/retry loop as
        # analyze_prospectus_comprehensive (see src/analyzers/llm_prospectus_analyzer.py),
        # reusing the financial_metrics/benchmarking/ipo_specifics already extracted above
        # rather than re-running the whole pipeline, so this costs exactly one extra
        # critique call plus one extra extraction call per retry the critique itself requests.
        t4 = time.time()
        if metrics is not None and benchmarking is not None and ipo_specifics is not None:
            try:
                reflection_attempts = 0
                reflection = analyzer._critique_extraction(
                    metrics, benchmarking, ipo_specifics, company_name,
                    attempt=0, max_attempts=MAX_REFLECTION_RETRIES
                )
                while reflection.ran and reflection.should_retry and reflection_attempts < MAX_REFLECTION_RETRIES:
                    reflection_attempts += 1
                    metrics = analyzer._extract_financial_metrics(
                        prospectus_text, company_name, feedback=reflection.issues
                    )
                    reflection = analyzer._critique_extraction(
                        metrics, benchmarking, ipo_specifics, company_name,
                        attempt=reflection_attempts, max_attempts=MAX_REFLECTION_RETRIES
                    )
                reflection.iterations_used = reflection_attempts
                result["reflection"] = to_plain(reflection)

                result["checks_after_reflection"] = score_financial_metrics(metrics, expected)
                result["raw_financial_metrics_after_reflection"] = to_plain(metrics)
                # "checks" stays the score reported/summarized by default - post-reflection,
                # since that's what analyze_prospectus_comprehensive actually returns now.
                result["checks"].extend(result["checks_after_reflection"])

                try:
                    verdict_check, extracted_verdict, truth_verdict = score_final_verdict(metrics, expected)
                    result["checks"].append(verdict_check)
                    result["final_verdict"] = {"extracted": extracted_verdict, "ground_truth": truth_verdict}
                except Exception as e:
                    result["errors"].append(f"score_final_verdict (after reflection) failed: {e}")
            except Exception as e:
                result["errors"].append(f"reflection loop failed: {e}")
        else:
            # Financial metrics, benchmarking, or IPO specifics extraction failed above -
            # fall back to scoring whatever financial metrics we do have, unreflected.
            if metrics is not None:
                result["checks"].extend(result.get("checks_before_reflection", []))
        result["timings_sec"]["reflection"] = round(time.time() - t4, 1)

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
        lines.append(f"**Score: {n_pass}/{n_total} checks passed** (post-reflection)")
        lines.append("")

        reflection = res.get("reflection")
        before = res.get("checks_before_reflection")
        after = res.get("checks_after_reflection")
        if reflection is not None and before is not None and after is not None:
            n_before = sum(1 for c in before if c["result"] == "PASS")
            n_after = sum(1 for c in after if c["result"] == "PASS")
            lines.append(
                f"**Self-critique/reflection: {reflection['iterations_used']} re-extraction "
                f"attempt(s) used** (cap: {MAX_REFLECTION_RETRIES}), "
                f"confidence={reflection['confidence']}"
            )
            lines.append(
                f"Financial-metric checks: {n_before}/{len(before)} before reflection -> "
                f"{n_after}/{len(after)} after reflection"
            )
            if reflection["issues"]:
                lines.append("Issues the critique pass flagged:")
                for issue in reflection["issues"]:
                    lines.append(f"- {issue}")
            before_by_name = {c["check"]: c["result"] for c in before}
            after_by_name = {c["check"]: c["result"] for c in after}
            flips = [
                f"- `{name}`: {before_by_name[name]} -> {after_by_name[name]}"
                for name in before_by_name
                if name in after_by_name and before_by_name[name] != after_by_name[name]
            ]
            if flips:
                lines.append("Checks that flipped due to reflection:")
                lines.extend(flips)
            fv_before = res.get("final_verdict_before_reflection")
            fv_after = res.get("final_verdict")
            if fv_before and fv_after:
                rec_before = fv_before["extracted"]["recommendation"]
                rec_after = fv_after["extracted"]["recommendation"]
                if rec_before != rec_after:
                    lines.append(
                        f"**Final verdict changed due to reflection: {rec_before} -> {rec_after}**"
                    )
            lines.append("")

        final_verdict = res.get("final_verdict")
        if final_verdict:
            ev, tv = final_verdict["extracted"], final_verdict["ground_truth"]
            match = "MATCH" if ev["recommendation"] == tv["recommendation"] else "MISMATCH"
            lines.append(
                f"**Final verdict: {match}** - extracted = {ev['recommendation']} "
                f"(score {ev['long_term_score']}/10) "
                f"vs. ground truth = {tv['recommendation']} "
                f"(score {tv['long_term_score']}/10)"
            )
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
