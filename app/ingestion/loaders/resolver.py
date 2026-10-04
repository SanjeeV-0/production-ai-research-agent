from pathlib import Path

from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PDFLoader

LOADERS_BY_SUFFIX: dict[str, type[DocumentLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
}


def resolve_loader_for_filename(filename: str) -> DocumentLoader:
    """Pick the registered loader implementation for a file, by extension.

    Shared by the upload HTTP route (for a freshly uploaded filename) and
    `IngestionService.retry_document` (for a previously stored filename), so
    there is exactly one filename-to-loader mapping, not two.
    """

    suffix = Path(filename).suffix.lower()

    loader_class = LOADERS_BY_SUFFIX.get(suffix)

    if loader_class is None:
        supported = ", ".join(sorted(LOADERS_BY_SUFFIX))
        raise ValueError(
            f"Unsupported file type '{suffix or filename}'. Supported types: {supported}."
        )

    return loader_class()
