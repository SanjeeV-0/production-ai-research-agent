from dataclasses import dataclass

from app.ingestion.structure import TableData
from app.ingestion.table_serializer import table_to_markdown


@dataclass(frozen=True)
class TableFragment:
    """A retrieval-sized fragment of a table."""

    table_id: str
    title: str | None
    fragment_index: int
    fragment_count: int
    content: str
    page_numbers: list[int]


def _estimate_tokens(text: str) -> int:
    """Estimate token count for deterministic chunk sizing."""
    return max(1, len(text.split()))


def split_table(
    table: TableData,
    max_tokens: int = 600,
) -> list[TableFragment]:
    """Split a large table into self-contained Markdown fragments."""
    if not table.rows:
        content = table_to_markdown(table)

        return [
            TableFragment(
                table_id=table.table_id,
                title=table.title,
                fragment_index=0,
                fragment_count=1,
                content=content,
                page_numbers=table.page_numbers,
            )
        ]

    fragments: list[list[list[str]]] = []
    current_rows: list[list[str]] = []

    for row in table.rows:
        candidate_rows = current_rows + [row]

        candidate = TableData(
            table_id=table.table_id,
            title=table.title,
            headers=table.headers,
            rows=candidate_rows,
            page_numbers=table.page_numbers,
        )

        if (
            current_rows
            and _estimate_tokens(table_to_markdown(candidate))
            > max_tokens
        ):
            fragments.append(current_rows)
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        fragments.append(current_rows)

    fragment_count = len(fragments)

    return [
        TableFragment(
            table_id=table.table_id,
            title=table.title,
            fragment_index=index,
            fragment_count=fragment_count,
            content=table_to_markdown(
                TableData(
                    table_id=table.table_id,
                    title=table.title,
                    headers=table.headers,
                    rows=rows,
                    page_numbers=table.page_numbers,
                )
            ),
            page_numbers=table.page_numbers,
        )
        for index, rows in enumerate(fragments)
    ]