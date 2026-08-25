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
    separator = "| " + " | ".join(
        "---" for _ in table.headers
    ) + " |"

    lines.append(header)
    lines.append(separator)

    for row in table.rows:
        normalized_row = list(row)

        if len(normalized_row) < len(table.headers):
            normalized_row.extend(
                [""] * (len(table.headers) - len(normalized_row))
            )

        normalized_row = normalized_row[: len(table.headers)]

        lines.append(
            "| " + " | ".join(normalized_row) + " |"
        )

    return "\n".join(lines).strip()