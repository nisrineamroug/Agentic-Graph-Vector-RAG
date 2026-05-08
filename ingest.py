# -*- coding: utf-8 -*-
"""
Main entry point for ingesting PDF data into both Vector and Graph RAG systems.
"""

import os
import sys
import argparse
from glob import glob
from typing import Optional

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.preprocessor import PDFPreprocessor
from backend.services.vectorial import VectorService
from backend.services.graph import GraphService
from backend.config import get_settings


def _resolve_pdf_path(settings, explicit_path: Optional[str]):
    if explicit_path:
        return explicit_path

    candidates = sorted(glob(os.path.join(settings.CORPUS_DIR, "*.pdf")))
    if candidates:
        return candidates[0]

    return os.path.join(settings.CORPUS_DIR, "16.pdf")

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the vector and graph pipelines.")
    parser.add_argument("--pdf", dest="pdf_path", default=None, help="Path to the PDF to ingest.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting text.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for splitting text.")
    parser.add_argument(
        "--save-cleaned",
        action="store_true",
        help="Save cleaned page-level text to data/cleaned_corpus/ by default.",
    )
    parser.add_argument(
        "--cleaned-output",
        default="",
        help="Explicit path for cleaned page-level JSON output.",
    )
    parser.add_argument(
        "--export-graph-json",
        action="store_true",
        help="Optionally export graph_store.json in addition to Neo4j sync.",
    )
    args = parser.parse_args()

    settings = get_settings()
    pdf_path = _resolve_pdf_path(settings, args.pdf_path)
    
    print("🚀 Starting Hybrid RAG Ingestion...")
    print(f"📄 Target: {pdf_path}")
    
    # --- STEP 1: Preprocessing ---
    print("\n--- Step 1: Preprocessing PDF ---")
    preprocessor = PDFPreprocessor(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    cleaned_output_path = args.cleaned_output
    if args.save_cleaned and not cleaned_output_path:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        cleaned_output_path = os.path.join("data", "cleaned_corpus", f"{base_name}_cleaned.json")

    try:
        chunks = preprocessor.run_pipeline(pdf_path, cleaned_output_path=cleaned_output_path)
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return

    # --- STEP 2: Vectorization ---
    print("\n--- Step 2: Building Vector Index (FAISS) ---")
    vector_service = VectorService()
    try:
        vector_service.create_index(chunks)
        vector_service.save_index()
    except Exception as e:
        print(f"❌ Error during vectorization: {e}")
        # We continue even if vector fails, maybe graph works?

    # --- STEP 3: Graph Construction ---
    print("\n--- Step 3: Building Knowledge Graph (Neo4j) ---")
    graph_service = GraphService()
    try:
        graph_service.create_graph_from_chunks(
            chunks,
            embeddings=vector_service.embeddings_cached,
            sync_to_neo4j=True,
            clear_existing=True,
            export_local_graph=args.export_graph_json,
        )
        print("✅ Graph ingestion complete.")
    except Exception as e:
        print(f"❌ Neo4j Error: {e}")
        print("💡 Tip: Check your .env file for correct Neo4j AuraDB credentials.")
        return
    finally:
        graph_service.close()

    print("\n🎉 ALL STEPS COMPLETE!")
    print("Next: You can now query your data using both Vector and Graph logic.")

if __name__ == "__main__":
    main()
