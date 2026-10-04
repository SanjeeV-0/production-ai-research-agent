"""Serializes a `TableData` into a standard Markdown pipe table.

Part of the active ingestion pipeline, used from three places: (1)
`app.ingestion.structure_extractor.StructureExtractor`, which re-serializes
every table it detects through this function so a TABLE unit's `content` has
canonical formatting regardless of source whitespace; (2)
`app.ingestion.loaders.docx.DocxLoader`, which renders python-docx's
structured table data as this same pipe-table syntax so `StructureExtractor`
can detect it; (3) `app.ingestion.table_chunker.split_table`, which
re-serializes each fragment. Also still reachable from the separate, unused
`structure_chunker.group_structural_units`.
"""

from app.ingestion.structure import TableData


def table_to_markdown(table: TableData) -> str:
    """Serialize a structured table into Markdown."""
    lines: list[str] = []

    if table.title:
        lines.append(f"### {table.title}")
        lines.append("")

    if not table.headers:
        return "\n".join(lines).strip()

    header = "| " + " | ".join(table.headers) + " |"
    separator = "| " + " | ".join("---" for _ in table.headers) + " |"

    lines.append(header)
    lines.append(separator)

    for row in table.rows:
        normalized_row = list(row)

        if len(normalized_row) < len(table.headers):
            normalized_row.extend([""] * (len(table.headers) - len(normalized_row)))

        normalized_row = normalized_row[: len(table.headers)]

        lines.append("| " + " | ".join(normalized_row) + " |")

    return "\n".join(lines).strip()
