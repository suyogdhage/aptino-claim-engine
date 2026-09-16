import os
import sys
from app.retrieval.hybrid_retriever import build_indices
from app.config import settings


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else settings.POLICY_PDF_PATH
    persist_dir = sys.argv[2] if len(sys.argv) > 2 else settings.CHROMA_PERSIST_DIR

    print(f"Ingesting {pdf_path} -> {persist_dir}")
    build_indices(pdf_path, persist_dir)
    print("Indexing complete.")


if __name__ == "__main__":
    main()