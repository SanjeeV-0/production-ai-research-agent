"""Scratch tool (not part of the test suite): runs the real production
ingestion steps per actual source format (Markdown / DOCX / PDF) for every
Benchmark V2 document and checks every gold fragment in questions_v2.json
against the REAL resulting chunk content -- not the raw source file.

Prints, per question, which fragments are MISSING, AMBIGUOUS, or OK.
"""

import json
from pathlib import Path
from uuid import uuid4

from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import shred_semantically
from app.ingestion.size_guard import apply_size_guard, fragment_table_units
from app.ingestion.structure_extractor import StructureExtractor

FIXTURES_DIR = Path(__file__).parent / "benchmark_v2"
MAX_CHUNK_TOKENS = 500

TITLE_TO_FILE_FORMAT: dict[str, tuple[str, str]] = {
    "Embedding Models for Semantic Retrieval": ("embeddings_for_retrieval.docx", "docx"),
    "Chunking Strategies for Retrieval-Augmented Generation": ("chunking_strategies.pdf", "pdf"),
    "PostgreSQL and pgvector for Production Vector Search": (
        "postgres_pgvector_operations.docx",
        "docx",
    ),
    "Architecture of a Production RAG Retrieval Pipeline": (
        "rag_pipeline_architecture.pdf",
        "pdf",
    ),
    "Query Decomposition in Practice": ("query_decomposition_practice.md", "md"),
    "Reranking and Candidate Pool Design": ("reranking_candidate_pool_design.md", "md"),
    "Evaluation Metrics and Benchmark Design for Retrieval Systems": (
        "retrieval_evaluation_metrics.docx",
        "docx",
    ),
    "Observability for Production Retrieval Systems": ("retrieval_observability.md", "md"),
    "Document Ingestion and Structural Extraction for RAG": (
        "document_ingestion_pipeline.pdf",
        "pdf",
    ),
}

LOADERS = {"md": MarkdownLoader(), "docx": DocxLoader(), "pdf": PDFLoader()}


def build_chunks_by_title(embedding_provider) -> dict[str, list[str]]:
    extractor = StructureExtractor()
    chunks_by_title: dict[str, list[str]] = {}

    for title, (filename, fmt) in TITLE_TO_FILE_FORMAT.items():
        path = FIXTURES_DIR / filename
        pages = LOADERS[fmt].load(path)
        structural_units = extractor.extract(pages)
        structural_units = fragment_table_units(structural_units, max_tokens=MAX_CHUNK_TOKENS)

        section_map = SectionMap()
        for unit in structural_units:
            if unit.section_path not in section_map._mapping:
                section_map.add(unit.section_path, uuid4())

        semantic_units = shred_semantically(
            structural_units,
            embedding_provider=embedding_provider,
            threshold=0.7,
        )
        child_chunks = apply_size_guard(
            semantic_units,
            section_map=section_map,
            max_tokens=MAX_CHUNK_TOKENS,
        )

        chunks_by_title[title] = [c.content for c in child_chunks]

    return chunks_by_title


def main() -> None:
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    chunks_by_title = build_chunks_by_title(embedding_provider)

    questions = json.loads(
        (FIXTURES_DIR / "questions_v2.json").read_text(encoding="utf-8"),
    )

    missing = []
    ambiguous = []
    ok_count = 0

    for q in questions:
        for rc in q["relevant_chunks"]:
            title, frag = rc["document_title"], rc["content_fragment"]
            contents = chunks_by_title.get(title)
            if contents is None:
                missing.append((q["id"], title, frag, "UNKNOWN DOC TITLE"))
                continue
            matches = [c for c in contents if frag in c]
            if len(matches) == 0:
                missing.append((q["id"], title, frag, "NOT FOUND"))
            elif len(matches) > 1:
                ambiguous.append((q["id"], title, frag, len(matches)))
            else:
                ok_count += 1

    total = sum(len(q["relevant_chunks"]) for q in questions)
    print(f"Total gold fragment references: {total}")
    print(f"OK: {ok_count}")
    print(f"Missing: {len(missing)}")
    print(f"Ambiguous: {len(ambiguous)}")
    print()
    print("--- MISSING (need new fragment) ---")
    for qid, title, frag, _reason in missing:
        print(f"{qid} [{title}]: {frag[:90]!r}")
    print()
    print("--- AMBIGUOUS (need more specific fragment) ---")
    for qid, title, frag, n in ambiguous:
        print(f"{qid} [{title}] x{n}: {frag[:90]!r}")


if __name__ == "__main__":
    main()
