from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agents.legal_rag.schemas import RetrievedLegalChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"

COLLECTION_NAME = "dpdp_legal_sources"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LegalRetriever:
    """Retrieve authoritative DPDP Act and Rules passages from ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
    ) -> None:
        self.persist_directory = Path(persist_directory)

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self.persist_directory}"
            )

        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
    ) -> list[RetrievedLegalChunk]:
        """Return the most relevant primary-source legal chunks."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        results = self.vector_store.similarity_search_with_score(
            query.strip(),
            k=k,
            filter={"authority": "primary"},
        )

        chunks = []

        for index, (document, distance) in enumerate(results, start=1):
            metadata = document.metadata

            chunks.append(
                RetrievedLegalChunk(
                    source_id=f"LEGAL-{metadata.get('source_file', 'unknown')}-"
                              f"P{metadata.get('pdf_page', 0)}-R{index}",
                    text=document.page_content,
                    source_file=metadata.get("source_file", "unknown"),
                    document_type=metadata.get("document_type", "unknown"),
                    authority=metadata.get("authority", "unknown"),
                    pdf_page=metadata.get("pdf_page", 1),
                    provision=metadata.get("provision", "not_identified"),
                    distance=float(distance),
                )
            )

        return chunks
