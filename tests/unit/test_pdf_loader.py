from pathlib import Path

from pypdf import PdfWriter

from app.ingestion.loaders.pdf import PDFLoader


def test_pdf_loader_reads_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    with pdf_path.open("wb") as file:
        writer.write(file)

    loader = PDFLoader()
    pages = loader.load(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].content == ""