from pathlib import Path

from pypdf import PdfReader

from app.ingestion.loaders.base import DocumentLoader, LoadedPage


class PDFLoader(DocumentLoader):
    """Load text from a PDF while preserving page boundaries."""

    def load(self, path: Path) -> list[LoadedPage]:
        reader = PdfReader(path)

        pages: list[LoadedPage] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                LoadedPage(
                    page_number=page_number,
                    content=text,
                )
            )

        return pages