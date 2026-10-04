from pathlib import Path

from docx import Document as DocxDocument

from app.ingestion.loaders.docx import DocxLoader


def test_docx_loader_extracts_paragraphs_in_document_order(tmp_path: Path) -> None:
    docx_path = tmp_path / "test.docx"

    document = DocxDocument()
    document.add_paragraph("Retrieval-Augmented Generation")
    document.add_paragraph("RAG combines retrieval with generation.")
    document.add_paragraph("It grounds answers in retrieved evidence.")
    document.save(docx_path)

    loader = DocxLoader()
    pages = loader.load(docx_path)

    assert len(pages) == 1
    assert pages[0].content == (
        "Retrieval-Augmented Generation\n"
        "RAG combines retrieval with generation.\n"
        "It grounds answers in retrieved evidence."
    )


def test_docx_loader_ignores_empty_paragraphs(tmp_path: Path) -> None:
    docx_path = tmp_path / "test.docx"

    document = DocxDocument()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("")
    document.add_paragraph("   ")
    document.add_paragraph("Second paragraph.")
    document.save(docx_path)

    loader = DocxLoader()
    pages = loader.load(docx_path)

    assert pages[0].content == "First paragraph.\nSecond paragraph."


def test_docx_loader_returns_page_number_one(tmp_path: Path) -> None:
    docx_path = tmp_path / "test.docx"

    document = DocxDocument()
    document.add_paragraph("Single page content.")
    document.save(docx_path)

    loader = DocxLoader()
    pages = loader.load(docx_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1


def test_docx_loader_returns_empty_content_for_document_with_no_text(tmp_path: Path) -> None:
    docx_path = tmp_path / "test.docx"

    document = DocxDocument()
    document.add_paragraph("")
    document.save(docx_path)

    loader = DocxLoader()
    pages = loader.load(docx_path)

    assert pages[0].page_number == 1
    assert pages[0].content == ""


def test_docx_loader_includes_table_text_after_paragraphs(tmp_path: Path) -> None:
    """Tables are supported via the simple, documented ordering: all
    paragraph text first, then each table's row text -- not interleaved at
    the table's true position in the document."""

    docx_path = tmp_path / "test.docx"

    document = DocxDocument()
    document.add_paragraph("Introduction paragraph.")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Header A"
    table.cell(0, 1).text = "Header B"
    table.cell(1, 0).text = "Value 1"
    table.cell(1, 1).text = "Value 2"

    document.save(docx_path)

    loader = DocxLoader()
    pages = loader.load(docx_path)

    assert pages[0].content == ("Introduction paragraph.\nHeader A | Header B\nValue 1 | Value 2")
