# -*- coding: utf-8 -*-
"""
Service for extracting text and images from PDFs and chunking the content.
"""

import os
import re
import unicodedata
import json
import fitz  # PyMuPDF
from typing import List, Dict, Any


class PDFPreprocessor:
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 400,
        remove_non_content_sections: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.remove_non_content_sections = remove_non_content_sections

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text

    def _extract_title_candidate(self, raw_text: str) -> str:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return ""

        # Ignore a leading line that is only a page number.
        first_line = lines[0]
        if first_line.isdigit() and len(lines) > 1:
            first_line = lines[1]

        # Keep the first meaningful words only to detect section headers.
        return " ".join(first_line.split()[:8])

    def _is_excluded_section_start(self, title_candidate: str) -> bool:
        normalized = self._normalize_text(title_candidate)
        excluded_pattern = r"^(remerciements?|dedicaces?|references?|bibliographie)\b"
        return bool(re.match(excluded_pattern, normalized))

    def _is_resume_section_start(self, title_candidate: str) -> bool:
        normalized = self._normalize_text(title_candidate)
        resume_pattern = (
            r"^(introduction|chapitre|chapter|conclusion|annexe|appendice|partie|resume|abstract)\b"
        )
        return bool(re.match(resume_pattern, normalized))

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from each page of the PDF.
        Returns a list of dictionaries containing page content and metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        pages_data = []
        doc = fitz.open(pdf_path)
        in_excluded_section = False

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text = page.get_text("text")
            title_candidate = self._extract_title_candidate(raw_text)

            if self.remove_non_content_sections:
                if self._is_excluded_section_start(title_candidate):
                    in_excluded_section = True
                elif in_excluded_section and self._is_resume_section_start(title_candidate):
                    in_excluded_section = False

                if in_excluded_section:
                    continue
            
            # Basic cleaning: remove extra whitespace
            clean_text = " ".join(raw_text.split())
            
            if clean_text:
                pages_data.append({
                    "page_number": page_num + 1,
                    "content": clean_text,
                    "image_count": len(page.get_images())
                })
        
        doc.close()
        return pages_data

    def create_chunks(self, pages_data: List[Dict[str, Any]], source_name: str = "unknown.pdf") -> List[Dict[str, Any]]:
        """
        Splits the extracted text into overlapping chunks using LangChain
        and extracts keywords using YAKE.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import yake
        
        # Initialize Splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize Keyword Extractor (Top 5 keywords)
        kw_extractor = yake.KeywordExtractor(lan="fr", n=2, dedupLim=0.9, top=5, features=None)

        chunks = []
        for page in pages_data:
            text = page["content"]
            page_num = page["page_number"]
            
            page_chunks = splitter.split_text(text)
            
            for chunk_text in page_chunks:
                # Extract keywords for this specific chunk
                keywords = kw_extractor.extract_keywords(chunk_text)
                keyword_list = [kw[0] for kw in keywords]
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "page_label": f"Page {page_num}",
                        "source": source_name,
                        "chunk_index": len(chunks),
                        "keywords": keyword_list  # Store as a list in metadata
                    }
                })
                
        return chunks

    def save_cleaned_pages(self, pages_data: List[Dict[str, Any]], output_path: str) -> str:
        """Persists cleaned page-level text to JSON for inspection/debugging."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)

        return output_path

    def run_pipeline(self, pdf_path: str, cleaned_output_path: str = ""):
        """
        Orchestrates the extraction and chunking process.
        """
        print(f"--- Starting Preprocessing for {pdf_path} ---")
        
        pages = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(pages)} pages.")

        if cleaned_output_path:
            saved_path = self.save_cleaned_pages(pages, cleaned_output_path)
            print(f"Saved cleaned pages to: {saved_path}")
        
        chunks = self.create_chunks(pages, source_name=os.path.basename(pdf_path))
        print(f"Created {len(chunks)} chunks from the text.")
        
        return chunks


# Example of how you might trigger this locally for testing:
if __name__ == "__main__":
    preprocessor = PDFPreprocessor()
    # Path to your 276-page PDF
    sample_path = os.path.join("data", "corpus", "16.pdf")
    try:
        results = preprocessor.run_pipeline(sample_path)
        print(f"First chunk preview: {results[0]['text'][:100]}...")
    except Exception as e:
        print(f"Error during processing: {e}")
