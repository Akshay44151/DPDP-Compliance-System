from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGAL_DOCS_DIR = PROJECT_ROOT / "knowledge_base" / "legal_docs"
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"

COLLECTION_NAME = "dpdp_legal_sources"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Only these official primary legal documents are ingested.
LEGAL_SOURCES = [
    {
        "filename": "2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "document_type": "DPDP Act, 2023",
        "authority": "primary",
        "pages": None,  # None means all pages.
    },
    {
        "filename": "DPDP_Rules_2025_English_only (1).pdf",
        "document_type": "DPDP Rules, 2025",
        "authority": "primary",
        # Zero-based PDF indexes: pages 1, 3, 5, ..., 17.
        "pages": list(range(0, 18, 2)),
    },
]


def extract_pdf_pages(source: dict) -> list[Document]:
    """Extract selected PDF pages and retain their original page numbers."""
    pdf_path = LEGAL_DOCS_DIR / source["filename"]

    if not pdf_path.exists():
        raise FileNotFoundError(f"Legal document not found: {pdf_path}")

    documents = []

    with pymupdf.open(pdf_path) as pdf:
        selected_pages = source["pages"]
        page_indexes = (
            selected_pages if selected_pages is not None
            else range(len(pdf))
        )

        for page_index in page_indexes:
            if page_index >= len(pdf):
                continue

            page = pdf[page_index]
            text = page.get_text("text").strip()

            if not text:
                print(
                    f"WARNING: No extractable text on "
                    f"{source['filename']}, page {page_index + 1}"
                )
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_file": source["filename"],
                        "document_type": source["document_type"],
                        "authority": source["authority"],
                        "pdf_page": page_index + 1,
                        "language": "English",
                    },
                )
            )

    return documents


def identify_provision(
    page_text: str,
    document_type: str,
    chunk_start: int,
) -> str:
    """Find the provision heading governing a chunk using its page position."""
    headings = list(re.finditer(
        r"(?m)^\s*(\d{1,2})\.\s*([^\n]{0,120})",
        page_text,
    ))

    preceding = [
        match for match in headings
        if match.start() <= chunk_start
    ]

    if preceding:
        match = preceding[-1]
    elif headings and headings[0].start() < 200:
        # Handle page headers that appear before the first provision heading.
        match = headings[0]
    else:
        return "not_identified"
    number = match.group(1)
    heading = match.group(2).strip()

    prefix = "Section" if "Act" in document_type else "Rule"
    return f"{prefix} {number}: {heading}"


def split_legal_pages(pages: list[Document]) -> list[Document]:
    """Split pages while preserving the correct provision heading."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_documents([page])

        for chunk in page_chunks:
            chunk.metadata["provision"] = identify_provision(
                page.page_content,
                page.metadata["document_type"],
                chunk.metadata.get("start_index", 0),
            )
            chunks.append(chunk)

    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def stable_chunk_id(document: Document, index: int) -> str:
    """Create a repeatable ID for each source page and chunk."""
    identity = (
        f"{document.metadata['source_file']}|"
        f"{document.metadata['pdf_page']}|"
        f"{document.metadata.get('start_index', 0)}|"
        f"{index}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ingest_legal_documents() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    total_chunks = 0

    for source in LEGAL_SOURCES:
        print(f"\nProcessing: {source['document_type']}")

        pages = extract_pdf_pages(source)
        chunks = split_legal_pages(pages)

        if not chunks:
            print("WARNING: No text chunks extracted.")
            continue

        # Replace only this source's previous chunks, making reruns safe.
        vector_store.delete(
            where={"source_file": source["filename"]}
        )

        ids = [
            stable_chunk_id(chunk, index)
            for index, chunk in enumerate(chunks)
        ]

        vector_store.add_documents(
            documents=chunks,
            ids=ids,
        )

        total_chunks += len(chunks)

        print(f"Pages extracted: {len(pages)}")
        print(f"Chunks stored: {len(chunks)}")

    print("\nLegal document ingestion completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"New chunks processed: {total_chunks}")
    print(f"Total vectors in collection: {vector_store._collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_legal_documents()
