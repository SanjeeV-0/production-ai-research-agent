from pathlib import Path

from docx import Document as DocxDocument

from app.ingestion.loaders.base import DocumentLoader, LoadedPage


class DocxLoader(DocumentLoader):
    """Load Microsoft Word (.docx) documents as a single logical page.

    Extracts body paragraph text in document order, skipping empty
    paragraphs, with paragraph boundaries preserved as newlines. Word
    documents don't carry reliable page-break metadata without rendering,
    so (like MarkdownLoader) no attempt is made to infer physical page
    numbers -- the whole document is returned as one LoadedPage.

    If the document contains tables, each non-empty row's cell text is
    appended -- as its own line, cells joined by " | " -- after all
    paragraph text, rather than interleaved at each table's true position.
    python-docx exposes `document.paragraphs` and `document.tables` as two
    separate flat sequences; reconstructing their true interleaved reading
    order would require walking the raw document XML, which is unnecessary
    complexity for this loader.
    """

    def load(self, path: Path) -> list[LoadedPage]:
        document = DocxDocument(path)

        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

        table_lines: list[str] = []
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    table_lines.append(" | ".join(cells))

        sections = ["\n".join(paragraphs)]
        if table_lines:
            sections.append("\n".join(table_lines))

        content = "\n".join(section for section in sections if section)

        return [LoadedPage(page_number=1, content=content)]
