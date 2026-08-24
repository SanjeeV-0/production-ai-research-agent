from pathlib import Path

from app.ingestion.loaders.base import DocumentLoader


class MarkdownLoader(DocumentLoader):
    """Load Markdown documents as UTF-8 text."""

    def load(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")