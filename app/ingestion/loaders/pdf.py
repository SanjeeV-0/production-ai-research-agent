from pathlib import Path

from pypdf import PdfReader

from app.ingestion.loaders.base import DocumentLoader


class PDFLoader(DocumentLoader):
    """Load text from a PDF document."""

    def load(self, path: Path) -> str:
        reader = PdfReader(path)

        pages: list[str] = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n\n".join(pages)