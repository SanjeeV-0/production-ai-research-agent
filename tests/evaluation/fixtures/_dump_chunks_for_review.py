"""Scratch tool (not part of the test suite): runs the real production
ingestion steps (loader -> StructureExtractor -> fragment_table_units ->
shred_semantically -> apply_size_guard) for every Benchmark V2 source
document and dumps the resulting chunk contents to text files, so gold
fragments can be chosen directly from real pipeline output rather than
guessed from source text.
"""

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
OUT_DIR = Path(
    r"C:\Users\SANJEE~1\AppData\Local\Temp\claude\C--Users-SanjeevSankarJ-Desktop-production-ai-research-agent"
    r"\baadbc5c-8426-4f09-8350-d8d1e9de89f9\scratchpad\chunks"
)

DOCS: list[tuple[str, str]] = [
    ("query_decomposition_practice.md", "md"),
    ("reranking_candidate_pool_design.md", "md"),
    ("retrieval_observability.md", "md"),
    ("embeddings_for_retrieval.docx", "docx"),
    ("postgres_pgvector_operations.docx", "docx"),
    ("retrieval_evaluation_metrics.docx", "docx"),
    ("chunking_strategies.pdf", "pdf"),
    ("rag_pipeline_architecture.pdf", "pdf"),
    ("document_ingestion_pipeline.pdf", "pdf"),
]

LOADERS = {"md": MarkdownLoader(), "docx": DocxLoader(), "pdf": PDFLoader()}
MAX_CHUNK_TOKENS = 500


def main() -> None:
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    extractor = StructureExtractor()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, fmt in DOCS:
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

        out_path = OUT_DIR / f"{filename}.chunks.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"FORMAT: {fmt}\nTOTAL CHUNKS: {len(child_chunks)}\n\n")
            for i, chunk in enumerate(child_chunks):
                f.write(f"===== CHUNK {i} ({len(chunk.content.split())} words) =====\n")
                f.write(chunk.content)
                f.write("\n\n")

        print(f"{filename}: {len(child_chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
