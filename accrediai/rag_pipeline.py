"""
AccrediAI – RAG Pipeline
========================
Modular Retrieval-Augmented Generation pipeline.

Architecture
------------
                  ┌─────────────────────────────────────┐
                  │           VectorStore (ABC)          │
                  │  – add_documents(chunks)             │
                  │  – similarity_search(query, k)       │
                  │  – rebuild(doc_meta_list)            │
                  │  – delete_store()                    │
                  └─────────────┬───────────────────────┘
                                │
                  ┌─────────────▼───────────────────────┐
                  │        FAISSVectorStore              │
                  │  (default implementation)            │
                  └─────────────────────────────────────┘

Swapping the vector store: pass a different `VectorStoreBase` subclass to
`RAGPipeline(vector_store=MyCustomStore(...))`.

Public API
----------
  pipeline = RAGPipeline()

  # Ingestion
  doc_meta = pipeline.ingest(file_path, original_filename)

  # Retrieval + generation
  answer, sources = pipeline.query(question, top_k=5)

  # Management
  pipeline.delete_document(doc_id)
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum characters a page must contain to be considered non-empty
_MIN_PAGE_CHARS = 30

# ---------------------------------------------------------------------------
# PDF extraction helpers  (3-tier fallback)
# ---------------------------------------------------------------------------

def _pdf_via_pypdf(file_path: str) -> list[Document]:
    """Tier 1 – pypdf (fast, text-layer only)."""
    return PyPDFLoader(file_path).load()


def _pdf_via_pdfplumber(file_path: str) -> list[Document]:
    """
    Tier 2 – pdfplumber (better layout reconstruction, extracts table text
    that pypdf misses, still text-layer only).
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return []

    docs: list[Document] = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also pull any tables as plain text rows
            for table in page.extract_tables():
                for row in table:
                    text += "\n" + "\t".join(cell or "" for cell in row)
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": os.path.basename(file_path), "page": i},
                ))
    return docs


def _pdf_via_pymupdf(file_path: str) -> list[Document]:
    """
    Tier 3 – PyMuPDF / fitz.
    Extracts text blocks with better Unicode handling.  Works on many PDFs
    that confuse pypdf (rotated pages, mixed encodings, complex fonts).
    Note: this is still text-layer extraction, not OCR.  Scanned image-only
    PDFs will still return empty pages.
    """
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore  (older package name)
        except ImportError:
            return []

    docs: list[Document] = []
    with pymupdf.open(file_path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text("text")  # plain UTF-8 text
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": os.path.basename(file_path), "page": i + 1},
                ))
    return docs


def _load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF using a 3-tier strategy:
      Tier 1 – PyPDF  (fastest, standard)
      Tier 2 – pdfplumber  (better layout / table handling)
      Tier 3 – PyMuPDF  (deepest text extraction, best Unicode)

    Each tier is tried only if the previous tier returned too little text.
    A page with fewer than _MIN_PAGE_CHARS characters is treated as empty.
    """
    def _non_empty(docs: list[Document]) -> list[Document]:
        return [d for d in docs if len(d.page_content.strip()) >= _MIN_PAGE_CHARS]

    # Tier 1
    try:
        pages = _non_empty(_pdf_via_pypdf(file_path))
        if pages:
            logger.info("PDF loaded via pypdf: %d pages with text", len(pages))
            return pages
    except Exception as e:
        logger.warning("pypdf failed (%s), trying pdfplumber…", e)

    # Tier 2
    try:
        pages = _non_empty(_pdf_via_pdfplumber(file_path))
        if pages:
            logger.info("PDF loaded via pdfplumber: %d pages with text", len(pages))
            return pages
    except Exception as e:
        logger.warning("pdfplumber failed (%s), trying PyMuPDF…", e)

    # Tier 3
    try:
        pages = _non_empty(_pdf_via_pymupdf(file_path))
        if pages:
            logger.info("PDF loaded via PyMuPDF: %d pages with text", len(pages))
            return pages
    except Exception as e:
        logger.warning("PyMuPDF failed (%s)", e)

    # All tiers exhausted
    logger.warning(
        "PDF '%s': all extraction methods returned no text. "
        "The file may be a scanned image-only PDF. "
        "OCR (e.g. Adobe Acrobat's 'Make Searchable') is required to extract text.",
        os.path.basename(file_path),
    )
    return []


# ---------------------------------------------------------------------------
# Document loader  (dispatches to _load_pdf for PDFs)
# ---------------------------------------------------------------------------

def _load_file(file_path: str) -> list[Document]:
    """
    Load a supported document and return a list of LangChain Documents.

    PDF strategy: 3-tier extraction (pypdf → pdfplumber → PyMuPDF).
    Other formats: single dedicated loader.
    """
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return _load_pdf(file_path)

    loaders = {
        "txt":  lambda p: TextLoader(p, encoding="utf-8").load(),
        "docx": lambda p: Docx2txtLoader(p).load(),
        "xlsx": lambda p: UnstructuredExcelLoader(p).load(),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: pdf, txt, docx, xlsx")
    return loaders[ext](file_path)


# ---------------------------------------------------------------------------
# Splitter helper
# ---------------------------------------------------------------------------

def _split(pages: list[Document]) -> list[Document]:
    """Split pages into smaller, overlapping chunks optimised for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph breaks
    )
    return splitter.split_documents(pages)


# ---------------------------------------------------------------------------
# Abstract vector store
# ---------------------------------------------------------------------------

class VectorStoreBase(ABC):
    """
    Swap-friendly vector store interface.
    Implement this class to use a different backend (e.g. Chroma, Qdrant, Pinecone).
    """

    @abstractmethod
    def add_documents(self, chunks: list[Document]) -> None:
        """Embed and add chunks to the store (create or upsert)."""

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        Return the k most similar documents with their distance scores.
        Lower score = more relevant for L2 distance (FAISS default).
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the store has been built and is queryable."""

    @abstractmethod
    def delete_store(self) -> None:
        """Delete the persisted store completely."""

    @abstractmethod
    def rebuild(self, doc_meta_list: list[dict]) -> None:
        """Rebuild the entire store from a list of doc-metadata dicts."""


# ---------------------------------------------------------------------------
# FAISS implementation
# ---------------------------------------------------------------------------

class FAISSVectorStore(VectorStoreBase):
    """
    FAISS-backed vector store persisted to disk.

    Parameters
    ----------
    index_path : Path
        Directory where the index is saved/loaded.
    embed_model : str
        HuggingFace sentence-transformer model name.
    """

    def __init__(self, index_path: Path, embed_model: str = EMBED_MODEL) -> None:
        self._index_path = index_path
        self._embed_model = embed_model
        self._embeddings: Optional[HuggingFaceEmbeddings] = None  # lazy

    # -- private helpers ---------------------------------------------------

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", self._embed_model)
            self._embeddings = HuggingFaceEmbeddings(model_name=self._embed_model)
        return self._embeddings

    def _load(self) -> Optional[FAISS]:
        if self._index_path.exists():
            return FAISS.load_local(
                str(self._index_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return None

    def _save(self, vs: FAISS) -> None:
        self._index_path.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self._index_path))
        logger.info("FAISS index saved: %s", self._index_path)

    # -- VectorStoreBase implementation ------------------------------------

    def add_documents(self, chunks: list[Document]) -> None:
        vs = self._load()
        if vs is None:
            vs = FAISS.from_documents(chunks, self._get_embeddings())
        else:
            vs.add_documents(chunks)
        self._save(vs)

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        vs = self._load()
        if vs is None:
            return []
        return vs.similarity_search_with_score(query, k=k)

    def is_ready(self) -> bool:
        return self._index_path.exists()

    def delete_store(self) -> None:
        if self._index_path.exists():
            shutil.rmtree(str(self._index_path))
            logger.info("FAISS index deleted: %s", self._index_path)

    def rebuild(self, doc_meta_list: list[dict]) -> None:
        self.delete_store()
        if not doc_meta_list:
            logger.info("No documents remain; index cleared.")
            return
        all_chunks: list[Document] = []
        for meta in doc_meta_list:
            fp = meta.get("file_path", "")
            if not os.path.exists(fp):
                logger.warning("File not found, skipping re-ingest: %s", fp)
                continue
            try:
                pages = _load_file(fp)
                for page in pages:
                    page.metadata["source"] = meta["filename"]
                all_chunks.extend(_split(pages))
            except Exception as exc:
                logger.warning("Could not re-ingest %s: %s", meta["filename"], exc)
        if all_chunks:
            vs = FAISS.from_documents(all_chunks, self._get_embeddings())
            self._save(vs)
            logger.info("FAISS index rebuilt: %d chunks from %d docs",
                        len(all_chunks), len(doc_meta_list))


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the full RAG workflow:
      upload → load → split → embed → store   (ingest)
      question → embed → retrieve → generate   (query)

    Parameters
    ----------
    vector_store : VectorStoreBase
        The backend to use for storage and retrieval.
        Defaults to FAISSVectorStore.
    """

    def __init__(self, vector_store: VectorStoreBase) -> None:
        self.store = vector_store

    # -- Ingestion ---------------------------------------------------------

    def ingest(self, file_path: str, original_filename: str) -> dict:
        """
        Load → split → tag metadata → embed → upsert.
        Returns a metadata dict describing the ingested document.
        """
        logger.info("Ingesting: %s", original_filename)
        pages = _load_file(file_path)

        if not pages:
            raise ValueError(f"Could not extract any content from '{original_filename}'.")

        # Tag every page with its source filename for citation
        for page in pages:
            page.metadata["source"] = original_filename

        chunks = _split(pages)
        if not chunks:
            raise ValueError(f"Document '{original_filename}' produced no text chunks.")

        logger.info("Chunks produced: %d (from %d pages)", len(chunks), len(pages))
        self.store.add_documents(chunks)

        return {
            "id": str(uuid.uuid4()),
            "filename": original_filename,
            "file_path": file_path,
            "pages": len(pages),
            "chunks": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
        }

    # -- Retrieval ---------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """
        Retrieve the top_k most relevant chunks for `question`.
        Returns (context_string, sources_list).

        context_string: formatted text ready to inject into the LLM prompt.
        sources_list:   list of dicts with document/page/excerpt/score for the UI.
        """
        if not self.store.is_ready():
            return "", []

        results = self.store.similarity_search(question, k=top_k)
        if not results:
            return "", []

        context_parts: list[str] = []
        sources: list[dict] = []
        seen: set[str] = set()

        for doc, score in results:
            source = doc.metadata.get("source", "Unknown document")
            raw_page = doc.metadata.get("page", None)
            # LangChain PDF pages are 0-indexed
            page_num = (raw_page + 1) if raw_page is not None else None
            page_label = f" – Page {page_num}" if page_num is not None else ""

            context_parts.append(
                f"[Source: {source}{page_label}]\n{doc.page_content.strip()}"
            )

            key = f"{source}{page_label}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": source,
                    "page": page_num,
                    "section": doc.metadata.get("section", None),
                    "excerpt": doc.page_content[:350].strip() + "…",
                    "relevance_score": round(float(score), 4),
                })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    # -- Delete ------------------------------------------------------------

    def delete_document(self, doc_meta_list: list[dict]) -> None:
        """Rebuild the index after a document has been removed from meta_list."""
        self.store.rebuild(doc_meta_list)
