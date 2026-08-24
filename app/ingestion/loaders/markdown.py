from pathlib import Path

from app.ingestion.loaders.base import DocumentLoader, LoadedPage


class MarkdownLoader(DocumentLoader):
    """Load Markdown documents as a single logical page."""

    def load(self, path: Path) -> list[LoadedPage]:
        return [
            LoadedPage(
                page_number=1,
                content=path.read_text(encoding="utf-8"),
            )
        ]