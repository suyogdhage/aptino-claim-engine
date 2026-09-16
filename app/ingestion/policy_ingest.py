import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pdfplumber


@dataclass
class PolicyChunk:
    chunk_id: str
    text: str
    page: int
    section: str
    subsection: Optional[str]
    clause_id: Optional[str]
    token_count: int


# Headings that structure the policy document (all-caps blocks).
# Reject short single words (AND, NOT, etc.) and city-name pairs from contacts/appendix.
KNOWN_SECTIONS = {
    "GENERAL", "DEFINITIONS", "SCOPE OF COVER", "WHAT WE COVER", "WHAT WE EXCLUDE",
    "CLAIMS PROCEDURE", "EXTENSIONS", "STANDARD TERMS AND CONDITIONS", "SCHEDULE",
    "SPECIFIC DISEASES", "CRITICAL ILLNESS COVER", "APPENDIX", "ANNEXURE",
}

def _is_heading(line: str) -> bool:
    if not re.match(r'^[A-Z][A-Z\s&/()\-.\']*$', line):
        return False
    if len(line) <= 3:
        return False
    if line in KNOWN_SECTIONS:
        return True
    # Attractive headings only when multi-word or reasonably long
    words = line.split()
    if len(words) >= 2 and all(w not in {"AND", "OR", "NOT"} for w in words):
        return len(line) >= 6
    return len(line) >= 12 and len(words) >= 2

SECTION_HEADING_PATTERN = None

# Definition entries: "Term means ..."
DEFINITION_PATTERN = re.compile(r'^([A-Z][A-Za-z\s/\-]+?)\s+means\s', re.IGNORECASE)

# Top-level numbered items inside WHAT WE COVER / WHAT WE EXCLUDE: "1. Room", "3. Hospitalization..."
NUMBERED_ITEM_PATTERN = re.compile(r'^(\d+)\.\s+([A-Z])')

# Exclusion / Exclusions keyword
EXCLUSION_PATTERN = re.compile(r'^Exclusion[s]?\s*$', re.IGNORECASE)

# Contacts / appendix markers: lines that indicate non-policy content
CONTACT_NOISE = re.compile(r'(Office of the Insurance Ombudsman|Tel\.:|Fax:|Email:|\.\.\.|\.\*\*|END)', re.IGNORECASE)

# City-name pairs from the ombudsman appendix (e.g. "AHMEDABAD BENGALURU")
CITY_HEADINGS = {
    "AHMEDABAD BENGALURU", "BHOPAL BHUBANESHWAR", "CHANDIGARH CHENNAI",
    "DELHI GUWAHATI", "ERNAKULAM KOLKATA", "HYDERABAD JAIPUR",
    "LUCKNOW MUMBAI", "NOIDA PATNA", "PUNE",
}


class PolicyIngestor:
    def __init__(
        self,
        pdf_path: str,
        chunk_size_tokens: int = 500,
        overlap_tokens: int = 50,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        policy_name: str = "USGIC-CSC-2017-2018"
    ):
        self.pdf_path = pdf_path
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_model_name = embedding_model
        self.policy_name = policy_name
        self._embedder = None
        self._global_counter = 0

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def extract_pages(self) -> List[Dict[str, Any]]:
        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append({"page": i + 1, "text": text})
        return pages

    def identify_sections(self, text: str, initial_section: Optional[str] = None,
                          initial_subsection: Optional[str] = None,
                          initial_clause: Optional[str] = None) -> List[Dict[str, Any]]:
        lines = text.split("\n")
        sections = []
        current_section = initial_section
        current_subsection = initial_subsection
        current_clause = initial_clause
        buffer = []

        def flush():
            nonlocal buffer, current_section, current_subsection, current_clause
            if buffer and (current_section or current_subsection or current_clause):
                sections.append({
                    "section": current_section,
                    "subsection": current_subsection,
                    "clause_id": current_clause,
                    "text": "\n".join(buffer),
                })
            buffer = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            # Drop page footers
            if re.search(r'IRDAI Reg No', line):
                continue
            if re.match(r'^CSC- Individual Health Insurance-Policy Wording', line):
                continue
            # Header line repeated on each page
            if line.startswith("UNIVERSAL SOMPO GENERAL INSURANCE CO LTD"):
                continue
            if line == "PROSPECTUS":
                continue
            # Non-policy contact/ombudsman content
            if CONTACT_NOISE.search(line):
                continue
            if line in CITY_HEADINGS:
                continue

            # Definition entries carry the term as a clause reference
            def_match = DEFINITION_PATTERN.match(line)
            if def_match and current_section == "DEFINITIONS":
                flush()
                current_clause = def_match.group(1).strip()
                buffer.append(line)
                continue

            # Section headings (all-caps standalone lines)
            if _is_heading(line) and len(line) <= 60:
                flush()
                current_section = line
                current_subsection = None
                current_clause = None
                buffer.append(line)
                continue

            # Exclusion keyword
            if EXCLUSION_PATTERN.match(line):
                flush()
                current_subsection = line
                current_clause = None
                buffer.append(line)
                continue

            # Numbered top-level items introduce a clause block
            if NUMBERED_ITEM_PATTERN.match(line):
                flush()
                current_clause = line
                buffer.append(line)
                continue

            buffer.append(line)

        flush()
        return sections

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.chunk_size_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size_tokens, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - self.overlap_tokens
        return chunks

    def _next_chunk_id(self, page: int, section: str) -> str:
        self._global_counter += 1
        content = f"{self.policy_name}|p{page}|{self._global_counter}|{section}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def ingest(self) -> List[PolicyChunk]:
        pages = self.extract_pages()
        all_chunks = []
        running_section = None
        running_subsection = None
        running_clause = None

        for page_data in pages:
            page = page_data["page"]
            sections = self.identify_sections(
                page_data["text"],
                initial_section=running_section,
                initial_subsection=running_subsection,
                initial_clause=running_clause
            )

            if not sections:
                sections = [{"section": running_section, "subsection": running_subsection,
                             "clause_id": running_clause, "text": page_data["text"]}]

            for sec in sections:
                running_section = sec["section"]
                running_subsection = sec["subsection"]
                running_clause = sec["clause_id"]

                sub_chunks = self.chunk_text(sec["text"])
                for chunk_text in sub_chunks:
                    all_chunks.append(PolicyChunk(
                        chunk_id=self._next_chunk_id(page, sec["section"] or "General"),
                        text=chunk_text,
                        page=page,
                        section=sec["section"] or "General",
                        subsection=sec["subsection"],
                        clause_id=sec["clause_id"],
                        token_count=len(chunk_text.split())
                    ))

        return all_chunks

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=True).tolist()


def ingest_policy(
    pdf_path: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[PolicyChunk]:
    ingestor = PolicyIngestor(pdf_path, chunk_size_tokens, overlap_tokens)
    return ingestor.ingest()