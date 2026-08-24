from pathlib import Path

from app.ingestion.loaders.markdown import MarkdownLoader


def test_markdown_loader_reads_utf8_text(tmp_path: Path) -> None:
    document = tmp_path / "test.md"
    document.write_text(
        "# RAG\n\nRetrieval-Augmented Generation.",
        encoding="utf-8",
    )

    loader = MarkdownLoader()

    assert loader.load(document) == (
        "# RAG\n\nRetrieval-Augmented Generation."
    )