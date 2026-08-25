from app.ingestion.structure import TableData
from app.ingestion.table_serializer import table_to_markdown


def test_table_to_markdown() -> None:
    table = TableData(
        table_id="table_1",
        title="Retrieval Performance",
        headers=["Model", "Recall@10", "MRR"],
        rows=[
            ["Model A", "0.82", "0.71"],
            ["Model B", "0.89", "0.78"],
        ],
        page_numbers=[10],
    )

    result = table_to_markdown(table)

    assert "### Retrieval Performance" in result
    assert "| Model | Recall@10 | MRR |" in result
    assert "| --- | --- | --- |" in result
    assert "| Model A | 0.82 | 0.71 |" in result
    assert "| Model B | 0.89 | 0.78 |" in result



def test_table_to_markdown_normalizes_short_rows() -> None:
    table = TableData(
        table_id="table_2",
        title=None,
        headers=["Model", "Score", "Rank"],
        rows=[
            ["Model A", "0.92"],
        ],
    )

    result = table_to_markdown(table)

    assert "| Model A | 0.92 |  |" in result