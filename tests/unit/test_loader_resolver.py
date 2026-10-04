import pytest

from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.loaders.resolver import resolve_loader_for_filename


def test_resolver_returns_markdown_loader_for_md() -> None:
    assert isinstance(resolve_loader_for_filename("paper.md"), MarkdownLoader)


def test_resolver_returns_markdown_loader_for_markdown_extension() -> None:
    assert isinstance(resolve_loader_for_filename("paper.markdown"), MarkdownLoader)


def test_resolver_returns_pdf_loader_for_pdf() -> None:
    assert isinstance(resolve_loader_for_filename("paper.pdf"), PDFLoader)


def test_resolver_returns_docx_loader_for_docx() -> None:
    assert isinstance(resolve_loader_for_filename("paper.docx"), DocxLoader)


def test_resolver_is_case_insensitive() -> None:
    assert isinstance(resolve_loader_for_filename("PAPER.DOCX"), DocxLoader)


def test_resolver_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        resolve_loader_for_filename("paper.exe")


def test_resolver_rejects_legacy_doc_extension() -> None:
    """Legacy .doc (pre-2007 binary Word format) is intentionally not
    supported -- only modern .docx."""

    with pytest.raises(ValueError, match="Unsupported file type"):
        resolve_loader_for_filename("paper.doc")
