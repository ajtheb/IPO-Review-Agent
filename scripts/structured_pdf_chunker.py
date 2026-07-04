#!/usr/bin/env python3
"""
Structured PDF Chunker for IPO Prospectus Documents

This script performs intelligent chunking of IPO prospectus PDFs based on:
- Document structure (sections, subsections)
- Topics and subtopics
- Semantic boundaries
- Metadata extraction

Designed specifically for financial documents like the Vidya Wires prospectus.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Installing required packages...")
    os.system("pip install PyPDF2 pdfplumber")
    import PyPDF2
    import pdfplumber

import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define chunk data structures at module level
@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: int
    section_title: str
    subsection_title: str
    page_start: int
    page_end: int
    pages: List[int]
    categories: List[str]
    has_tables: bool
    has_numbers: bool
    char_count: int
    word_count: int
    content_type: str


@dataclass
class Chunk:
    """A document chunk with metadata and content."""
    metadata: ChunkMetadata
    content: str
    tables: List[Dict] = field(default_factory=list)


class StructuredPDFChunker:
    """
    Intelligent chunker for IPO prospectus documents.
    Identifies structure and creates semantically meaningful chunks.
    """
    
    # Common prospectus section patterns
    SECTION_PATTERNS = [
        # Main sections (all caps or title case with numbers)
        r'^[A-Z\s]{10,}$',  # ALL CAPS SECTIONS
        r'^\d+\.\s+[A-Z][A-Za-z\s]+$',  # 1. Title Case Section
        r'^[IVX]+\.\s+[A-Z][A-Za-z\s]+$',  # Roman numerals
        
        # Common prospectus sections
        r'^(RISK FACTORS|ABOUT THE COMPANY|BUSINESS OVERVIEW|MANAGEMENT|FINANCIAL)',
        r'^(OBJECTS OF THE OFFER|USE OF PROCEEDS|CAPITAL STRUCTURE|INDUSTRY)',
        r'^(REGULATIONS|BOARD OF DIRECTORS|DIVIDEND|OUTSTANDING LITIGATION)',
        r'^(RELATED PARTY|SUMMARY|BASIS|GENERAL INFORMATION|ISSUE)',
    ]
    
    # Subsection patterns
    SUBSECTION_PATTERNS = [
        r'^\d+\.\d+\s+[A-Z]',  # 1.1 Subsection
        r'^[a-z]\)\s+',  # a) subsection
        r'^\([a-z]\)\s+',  # (a) subsection
        r'^\([ivx]+\)\s+',  # (i) subsection
    ]
    
    # Financial statement indicators
    FINANCIAL_PATTERNS = [
        r'(Balance Sheet|Income Statement|Cash Flow|Profit.*Loss)',
        r'(Financial Performance|Financial Position|Statement of)',
        r'(Assets|Liabilities|Revenue|Expenses|EBITDA|PAT|EPS)',
    ]
    
    def __init__(self, pdf_path: str, output_dir: str = "./structured_chunks"):
        """
        Initialize the chunker.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save structured chunks
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunks = []
        self.chunk_counter = 0  # Add chunk counter attribute
        self.metadata = {
            "file_name": self.pdf_path.name,
            "processing_date": datetime.now().isoformat(),
            "total_pages": 0,
            "total_chunks": 0,
            "structure": {}
        }
        
        logger.info(f"Initialized chunker for: {self.pdf_path}")
    
    def extract_text_with_structure(self) -> List[Dict]:
        """
        Extract text from PDF while preserving structure.
        
        Returns:
            List of page dictionaries with text and metadata
        """
        pages = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.metadata["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout
                    text = page.extract_text() or ""
                    
                    # Extract tables
                    tables = page.extract_tables() or []
                    
                    # Get page dimensions
                    width = page.width
                    height = page.height
                    
                    page_data = {
                        "page_number": page_num,
                        "text": text,
                        "tables": tables,
                        "dimensions": {"width": width, "height": height},
                        "has_tables": len(tables) > 0,
                        "char_count": len(text),
                        "line_count": len(text.split('\n'))
                    }
                    
                    pages.append(page_data)
                    
                    if page_num % 10 == 0:
                        logger.info(f"Processed {page_num}/{len(pdf.pages)} pages")
        
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            # Fallback to PyPDF2
            logger.info("Falling back to PyPDF2...")
            pages = self._extract_with_pypdf2()
        
        logger.info(f"Extracted text from {len(pages)} pages")
        return pages
    
    def _extract_with_pypdf2(self) -> List[Dict]:
        """Fallback extraction using PyPDF2."""
        pages = []
        
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.metadata["total_pages"] = len(pdf_reader.pages)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text() or ""
                    
                    page_data = {
                        "page_number": page_num + 1,
                        "text": text,
                        "tables": [],
                        "dimensions": {},
                        "has_tables": False,
                        "char_count": len(text),
                        "line_count": len(text.split('\n'))
                    }
                    
                    pages.append(page_data)
        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
        
        return pages
    
    def identify_section(self, text: str) -> Optional[Dict]:
        """
        Identify if text line is a section heading.
        
        Args:
            text: Text line to analyze
            
        Returns:
            Section metadata if found, None otherwise
        """
        text = text.strip()
        
        if not text or len(text) < 3:
            return None
        
        # Check against section patterns
        for pattern in self.SECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    "type": "section",
                    "title": text,
                    "level": 1,
                    "pattern_matched": pattern
                }
        
        # Check subsection patterns
        for pattern in self.SUBSECTION_PATTERNS:
            if re.search(pattern, text):
                return {
                    "type": "subsection",
                    "title": text,
                    "level": 2,
                    "pattern_matched": pattern
                }
        
        return None
    
    def classify_content(self, text: str) -> str:
        """
        Classify the type of content.
        
        Args:
            text: Text to classify
            
        Returns:
            Content category
        """
        text_lower = text.lower()
        
        # Financial content
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.FINANCIAL_PATTERNS):
            return "financial"
        
        # Risk factors
        if "risk" in text_lower and ("factor" in text_lower or "warning" in text_lower):
            return "risk"
        
        # Business/operations
        if any(word in text_lower for word in ["business", "operation", "product", "service", "manufacturing"]):
            return "business"
        
        # Management/governance
        if any(word in text_lower for word in ["director", "management", "board", "governance", "committee"]):
            return "management"
        
        # Legal/regulatory
        if any(word in text_lower for word in ["legal", "regulatory", "compliance", "litigation", "law"]):
            return "legal"
        
        # Market/industry
        if any(word in text_lower for word in ["market", "industry", "competition", "sector", "trend"]):
            return "market"
        
        # Default
        return "general"
    
    def create_structured_chunks(self, pages: List[Dict]) -> List[Dict]:
        """
        Create structured chunks from extracted pages.
        
        Args:
            pages: List of page data dictionaries
            
        Returns:
            List of structured chunks with metadata
        """
        chunks = []
        current_section = "Document Start"
        current_subsection = None
        current_chunk = []
        current_chunk_metadata = {
            "section": current_section,
            "subsection": current_subsection,
            "pages": [],
            "categories": set(),
            "has_tables": False,
            "has_numbers": False
        }
        
        chunk_id = 0
        
        for page_data in pages:
            page_num = page_data["page_number"]
            text = page_data["text"]
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    continue
                
                # Check if this is a section heading
                section_info = self.identify_section(line)
                
                if section_info:
                    # Save current chunk if it has content
                    if current_chunk:
                        chunk_id += 1
                        chunk_text = '\n'.join(current_chunk)
                        
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "metadata": {
                                "section": current_section,
                                "subsection": current_subsection,
                                "pages": list(set(current_chunk_metadata["pages"])),
                                "categories": list(current_chunk_metadata["categories"]),
                                "has_tables": current_chunk_metadata["has_tables"],
                                "has_numbers": current_chunk_metadata["has_numbers"],
                                "char_count": len(chunk_text),
                                "line_count": len(current_chunk),
                                "content_type": self.classify_content(chunk_text)
                            }
                        })
                    
                    # Start new chunk
                    if section_info["type"] == "section":
                        current_section = section_info["title"]
                        current_subsection = None
                    else:
                        current_subsection = section_info["title"]
                    
                    current_chunk = [line]
                    current_chunk_metadata = {
                        "section": current_section,
                        "subsection": current_subsection,
                        "pages": [page_num],
                        "categories": set(),
                        "has_tables": page_data["has_tables"],
                        "has_numbers": bool(re.search(r'\d', line))
                    }
                else:
                    # Add to current chunk
                    current_chunk.append(line)
                    current_chunk_metadata["pages"].append(page_num)
                    
                    # Update metadata
                    content_category = self.classify_content(line)
                    current_chunk_metadata["categories"].add(content_category)
                    
                    if page_data["has_tables"]:
                        current_chunk_metadata["has_tables"] = True
                    
                    if re.search(r'\d', line):
                        current_chunk_metadata["has_numbers"] = True
                    
                    # Check if chunk is getting too large (split at reasonable boundaries)
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) > 3000:  # Larger chunks for structured content
                        chunk_id += 1
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "metadata": {
                                "section": current_section,
                                "subsection": current_subsection,
                                "pages": list(set(current_chunk_metadata["pages"])),
                                "categories": list(current_chunk_metadata["categories"]),
                                "has_tables": current_chunk_metadata["has_tables"],
                                "has_numbers": current_chunk_metadata["has_numbers"],
                                "char_count": len(chunk_text),
                                "line_count": len(current_chunk),
                                "content_type": self.classify_content(chunk_text)
                            }
                        })
                        
                        # Start new chunk (with overlap)
                        overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                        current_chunk = overlap_lines
                        current_chunk_metadata["pages"] = [page_num]
        
        # Save final chunk
        if current_chunk:
            chunk_id += 1
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "section": current_section,
                    "subsection": current_subsection,
                    "pages": list(set(current_chunk_metadata["pages"])),
                    "categories": list(current_chunk_metadata["categories"]),
                    "has_tables": current_chunk_metadata["has_tables"],
                    "has_numbers": current_chunk_metadata["has_numbers"],
                    "char_count": len(chunk_text),
                    "line_count": len(current_chunk),
                    "content_type": self.classify_content(chunk_text)
                }
            })
        
        logger.info(f"Created {len(chunks)} structured chunks")
        return chunks
    
    def save_chunks(self, chunks: List[Dict]):
        """
        Save chunks to disk with metadata.
        
        Args:
            chunks: List of chunk dictionaries
        """
        # Create output directory structure
        company_name = self.pdf_path.stem.replace('_', ' ').title()
        output_path = self.output_dir / company_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual chunks
        chunks_dir = output_path / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        for chunk in chunks:
            chunk_file = chunks_dir / f"chunk_{chunk['chunk_id']:04d}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        # Save complete chunks file
        all_chunks_file = output_path / "all_chunks.json"
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Create structure index
        structure = self._create_structure_index(chunks)
        structure_file = output_path / "structure_index.json"
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        self.metadata["total_chunks"] = len(chunks)
        self.metadata["structure"] = structure
        self.metadata["output_path"] = str(output_path)
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        self._create_summary_report(chunks, output_path)
        
        logger.info(f"Saved {len(chunks)} chunks to: {output_path}")
        logger.info(f"Structure index: {structure_file}")
        logger.info(f"Metadata: {metadata_file}")
    
    def _create_structure_index(self, chunks: List[Dict]) -> Dict:
        """Create hierarchical structure index."""
        structure = {}
        
        for chunk in chunks:
            section = chunk["metadata"]["section"]
            subsection = chunk["metadata"].get("subsection")
            
            if section not in structure:
                structure[section] = {
                    "chunk_ids": [],
                    "subsections": {},
                    "pages": set(),
                    "categories": set()
                }
            
            structure[section]["chunk_ids"].append(chunk["chunk_id"])
            structure[section]["pages"].update(chunk["metadata"]["pages"])
            structure[section]["categories"].update(chunk["metadata"]["categories"])
            
            if subsection:
                if subsection not in structure[section]["subsections"]:
                    structure[section]["subsections"][subsection] = {
                        "chunk_ids": [],
                        "pages": set()
                    }
                
                structure[section]["subsections"][subsection]["chunk_ids"].append(chunk["chunk_id"])
                structure[section]["subsections"][subsection]["pages"].update(chunk["metadata"]["pages"])
        
        # Convert sets to lists for JSON serialization
        for section in structure:
            structure[section]["pages"] = sorted(list(structure[section]["pages"]))
            structure[section]["categories"] = list(structure[section]["categories"])
            
            for subsection in structure[section]["subsections"]:
                structure[section]["subsections"][subsection]["pages"] = sorted(
                    list(structure[section]["subsections"][subsection]["pages"])
                )
        
        return structure
    
    def _create_summary_report(self, chunks: List[Dict], output_path: Path):
        """Create human-readable summary report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STRUCTURED CHUNKING SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nDocument: {self.pdf_path.name}")
        report_lines.append(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Pages: {self.metadata['total_pages']}")
        report_lines.append(f"Total Chunks: {len(chunks)}")
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DOCUMENT STRUCTURE")
        report_lines.append("=" * 80)
        
        # Group by section
        sections = {}
        for chunk in chunks:
            section = chunk["metadata"]["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk)
        
        for section, section_chunks in sections.items():
            report_lines.append(f"\n📁 {section}")
            report_lines.append(f"   Chunks: {len(section_chunks)}")
            
            # Get unique pages
            pages = set()
            for chunk in section_chunks:
                pages.update(chunk["metadata"]["pages"])
            report_lines.append(f"   Pages: {sorted(list(pages))}")
            
            # Get categories
            categories = set()
            for chunk in section_chunks:
                categories.update(chunk["metadata"]["categories"])
            report_lines.append(f"   Categories: {', '.join(sorted(list(categories)))}")
            
            # Show subsections
            subsections = {}
            for chunk in section_chunks:
                subsection = chunk["metadata"].get("subsection")
                if subsection:
                    if subsection not in subsections:
                        subsections[subsection] = []
                    subsections[subsection].append(chunk)
            
            if subsections:
                for subsection, sub_chunks in subsections.items():
                    report_lines.append(f"   └── {subsection}")
                    report_lines.append(f"       Chunks: {len(sub_chunks)}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("CONTENT BREAKDOWN")
        report_lines.append("=" * 80)
        
        # Count by content type
        content_types = {}
        for chunk in chunks:
            content_type = chunk["metadata"]["content_type"]
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(chunks)) * 100
            report_lines.append(f"{content_type.title():15} : {count:3} chunks ({percentage:5.1f}%)")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("STATISTICS")
        report_lines.append("=" * 80)
        
        total_chars = sum(chunk["metadata"]["char_count"] for chunk in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        
        chunks_with_tables = sum(1 for chunk in chunks if chunk["metadata"]["has_tables"])
        chunks_with_numbers = sum(1 for chunk in chunks if chunk["metadata"]["has_numbers"])
        
        report_lines.append(f"Total Characters: {total_chars:,}")
        report_lines.append(f"Average Chunk Size: {avg_chars:.0f} characters")
        report_lines.append(f"Chunks with Tables: {chunks_with_tables} ({chunks_with_tables/len(chunks)*100:.1f}%)")
        report_lines.append(f"Chunks with Numbers: {chunks_with_numbers} ({chunks_with_numbers/len(chunks)*100:.1f}%)")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("OUTPUT FILES")
        report_lines.append("=" * 80)
        report_lines.append(f"📂 Output Directory: {output_path}")
        report_lines.append(f"📄 All Chunks: all_chunks.json")
        report_lines.append(f"📄 Structure Index: structure_index.json")
        report_lines.append(f"📄 Metadata: metadata.json")
        report_lines.append(f"📁 Individual Chunks: chunks/")
        report_lines.append("=" * 80)
        
        # Save report
        report_file = output_path / "summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to: {report_file}")
    
    def process_document(self, min_chunk_size: int = 200, max_chunk_size: int = 2000):
        """
        Main processing pipeline with configurable chunk sizes.
        This method is called from the Streamlit app.
        
        Args:
            min_chunk_size: Minimum characters per chunk (not used in current implementation)
            max_chunk_size: Maximum characters per chunk (used to split large chunks)
        """
        logger.info(f"Starting structured chunking process (max_chunk_size={max_chunk_size})...")
        
        # Extract text with structure
        pages = self.extract_text_with_structure()
        
        if not pages:
            logger.error("No pages extracted!")
            return
        
        # Create structured chunks with size limit
        self.chunks = self.create_structured_chunks_with_size(pages, max_chunk_size)
        
        if not self.chunks:
            logger.error("No chunks created!")
            return
        
        # Update chunk counter
        self.chunk_counter = len(self.chunks)
        self.metadata["total_chunks"] = self.chunk_counter
        
        logger.info(f"✅ Created {self.chunk_counter} structured chunks")
    
    def create_structured_chunks_with_size(self, pages: List[Dict], max_size: int) -> List:
        """
        Create structured chunks with size constraints.
        Returns list of chunk objects compatible with app expectations.
        """
        chunk_dicts = self.create_structured_chunks(pages)
        
        # Convert to chunk objects with proper structure
        chunks = []
        for chunk_dict in chunk_dicts:
            pages_list = chunk_dict["metadata"]["pages"]
            metadata = ChunkMetadata(
                chunk_id=chunk_dict["chunk_id"],
                section_title=chunk_dict["metadata"]["section"],
                subsection_title=chunk_dict["metadata"].get("subsection", ""),
                page_start=min(pages_list) if pages_list else 0,
                page_end=max(pages_list) if pages_list else 0,
                pages=pages_list,
                categories=chunk_dict["metadata"]["categories"],
                has_tables=chunk_dict["metadata"]["has_tables"],
                has_numbers=chunk_dict["metadata"]["has_numbers"],
                char_count=chunk_dict["metadata"]["char_count"],
                word_count=len(chunk_dict["text"].split()),
                content_type=chunk_dict["metadata"]["content_type"]
            )
            
            chunk = Chunk(
                metadata=metadata,
                content=chunk_dict["text"],
                tables=[]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def save_chunks(self, save_individual: bool = True):
        """
        Save chunks and metadata to files.
        Compatible with app expectations.
        
        Args:
            save_individual: Whether to save individual chunk files
        """
        if not self.chunks:
            logger.warning("No chunks to save!")
            return
        
        # Convert chunk objects back to dicts for saving
        chunks_as_dicts = []
        for chunk in self.chunks:
            chunk_dict = {
                "chunk_id": chunk.metadata.chunk_id,
                "text": chunk.content,
                "metadata": {
                    "section": chunk.metadata.section_title,
                    "subsection": chunk.metadata.subsection_title,
                    "pages": chunk.metadata.pages,
                    "categories": chunk.metadata.categories,
                    "has_tables": chunk.metadata.has_tables,
                    "has_numbers": chunk.metadata.has_numbers,
                    "char_count": chunk.metadata.char_count,
                    "line_count": len(chunk.content.split('\n')),
                    "content_type": chunk.metadata.content_type
                }
            }
            chunks_as_dicts.append(chunk_dict)
        
        # Save all chunks as JSON
        all_chunks_file = self.output_dir / "all_chunks.json"
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_as_dicts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved all chunks to: {all_chunks_file}")
        
        # Save individual chunks if requested
        if save_individual:
            chunks_dir = self.output_dir / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            for chunk_dict in chunks_as_dicts:
                chunk_file = chunks_dir / f"chunk_{chunk_dict['chunk_id']:04d}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks_as_dicts)} individual chunks to: {chunks_dir}")
        
        # Save structure index
        structure = self._create_structure_index(chunks_as_dicts)
        structure_file = self.output_dir / "structure_index.json"
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved structure index to: {structure_file}")
        
        # Save metadata
        self.metadata["structure"] = structure
        self.metadata["total_chunks"] = len(chunks_as_dicts)
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_file}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a human-readable summary report.
        Returns the report as a string.
        """
        if not self.chunks:
            return "No chunks available to generate report."
        
        # Convert chunks back to dicts for report generation
        chunks_as_dicts = []
        for chunk in self.chunks:
            chunk_dict = {
                "chunk_id": chunk.metadata.chunk_id,
                "text": chunk.content,
                "metadata": {
                    "section": chunk.metadata.section_title,
                    "subsection": chunk.metadata.subsection_title,
                    "pages": chunk.metadata.pages,
                    "categories": chunk.metadata.categories,
                    "has_tables": chunk.metadata.has_tables,
                    "has_numbers": chunk.metadata.has_numbers,
                    "char_count": chunk.metadata.char_count,
                    "line_count": len(chunk.content.split('\n')),
                    "content_type": chunk.metadata.content_type
                }
            }
            chunks_as_dicts.append(chunk_dict)
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STRUCTURED CHUNKING SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nDocument: {self.pdf_path.name}")
        report_lines.append(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Pages: {self.metadata['total_pages']}")
        report_lines.append(f"Total Chunks: {len(chunks_as_dicts)}")
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DOCUMENT STRUCTURE")
        report_lines.append("=" * 80)
        
        # Group by section
        sections = {}
        for chunk in chunks_as_dicts:
            section = chunk["metadata"]["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk)
        
        for section, section_chunks in list(sections.items())[:20]:  # Limit to first 20 sections
            report_lines.append(f"\n📁 {section}")
            report_lines.append(f"   Chunks: {len(section_chunks)}")
            
            # Get unique pages
            pages = set()
            for chunk in section_chunks:
                pages.update(chunk["metadata"]["pages"])
            report_lines.append(f"   Pages: {sorted(list(pages))[:10]}")  # Show first 10 pages
            
            # Get categories
            categories = set()
            for chunk in section_chunks:
                categories.update(chunk["metadata"]["categories"])
            report_lines.append(f"   Categories: {', '.join(sorted(list(categories)))}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("CONTENT BREAKDOWN")
        report_lines.append("=" * 80)
        
        # Count by content type
        content_types = {}
        for chunk in chunks_as_dicts:
            content_type = chunk["metadata"]["content_type"]
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(chunks_as_dicts)) * 100
            report_lines.append(f"{content_type.title():15} : {count:3} chunks ({percentage:5.1f}%)")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("STATISTICS")
        report_lines.append("=" * 80)
        
        total_chars = sum(chunk["metadata"]["char_count"] for chunk in chunks_as_dicts)
        avg_chars = total_chars / len(chunks_as_dicts) if chunks_as_dicts else 0
        
        chunks_with_tables = sum(1 for chunk in chunks_as_dicts if chunk["metadata"]["has_tables"])
        chunks_with_numbers = sum(1 for chunk in chunks_as_dicts if chunk["metadata"]["has_numbers"])
        
        report_lines.append(f"Total Characters: {total_chars:,}")
        report_lines.append(f"Average Chunk Size: {avg_chars:.0f} characters")
        report_lines.append(f"Chunks with Tables: {chunks_with_tables} ({chunks_with_tables/len(chunks_as_dicts)*100:.1f}%)")
        report_lines.append(f"Chunks with Numbers: {chunks_with_numbers} ({chunks_with_numbers/len(chunks_as_dicts)*100:.1f}%)")
        
        report_lines.append("\n" + "=" * 80)
        
        return '\n'.join(report_lines)
    
    def save_summary_report(self):
        """
        Generate and save the summary report to a file.
        """
        report = self.generate_summary_report()
        report_file = self.output_dir / "summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to: {report_file}")
    
    def process(self):
        """Main processing pipeline."""
        logger.info("Starting structured chunking process...")
        
        # Extract text with structure
        pages = self.extract_text_with_structure()
        
        if not pages:
            logger.error("No pages extracted!")
            return
        
        # Create structured chunks
        chunks = self.create_structured_chunks(pages)
        
        if not chunks:
            logger.error("No chunks created!")
            return
        
        # Save chunks with metadata (as dicts, not objects)
        self._save_chunks_internal(chunks)
        
        logger.info("✅ Structured chunking complete!")
        
        return chunks
    
    def _save_chunks_internal(self, chunks: List[Dict]):
        """Internal method to save chunks as dicts."""
        # Create output directory structure
        company_name = self.pdf_path.stem.replace('_', ' ').title()
        output_path = self.output_dir / company_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual chunks
        chunks_dir = output_path / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        for chunk in chunks:
            chunk_file = chunks_dir / f"chunk_{chunk['chunk_id']:04d}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        # Save complete chunks file
        all_chunks_file = output_path / "all_chunks.json"
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Create structure index
        structure = self._create_structure_index(chunks)
        structure_file = output_path / "structure_index.json"
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        self.metadata["total_chunks"] = len(chunks)
        self.metadata["structure"] = structure
        self.metadata["output_path"] = str(output_path)
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        self._create_summary_report(chunks, output_path)
        
        logger.info(f"Saved {len(chunks)} chunks to: {output_path}")
        logger.info(f"Structure index: {structure_file}")
        logger.info(f"Metadata: {metadata_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Structured PDF Chunker for IPO Prospectus Documents"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="vidya_wires.pdf",
        help="Path to PDF file (default: vidya_wires.pdf)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./structured_chunks",
        help="Output directory (default: ./structured_chunks)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        logger.info("Please provide a valid PDF path")
        return 1
    
    # Process the PDF
    chunker = StructuredPDFChunker(str(pdf_path), args.output)
    chunks = chunker.process()
    
    if chunks:
        logger.info(f"\n✅ Successfully processed {pdf_path.name}")
        logger.info(f"📊 Created {len(chunks)} structured chunks")
        logger.info(f"📁 Output: {chunker.output_dir / pdf_path.stem.replace('_', ' ').title()}")
        return 0
    else:
        logger.error("Failed to process PDF")
        return 1


if __name__ == "__main__":
    sys.exit(main())
