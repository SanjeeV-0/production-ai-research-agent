from app.ingestion.structure import TableData
from app.ingestion.table_chunker import split_table


def test_small_table_stays_as_one_fragment() -> None:
    table = TableData(
        table_id="table_1",
        title="Results",
        headers=["Model", "Score"],
        rows=[
            ["A", "0.90"],
            ["B", "0.85"],
        ],
        page_numbers=[10],
    )

    fragments = split_table(table, max_tokens=100)

    assert len(fragments) == 1
    assert fragments[0].fragment_index == 0
    assert fragments[0].fragment_count == 1
    assert "| Model | Score |" in fragments[0].content


def test_large_table_repeats_headers() -> None:
    table = TableData(
        table_id="table_2",
        title="Large Results",
        headers=["Model", "Score"],
        rows=[
            ["A", "0.90"],
            ["B", "0.85"],
            ["C", "0.82"],
            ["D", "0.80"],
            ["E", "0.78"],
        ],
        page_numbers=[10, 11],
    )

    fragments = split_table(table, max_tokens=15)

    assert len(fragments) > 1

    for fragment in fragments:
        assert "| Model | Score |" in fragment.content
        assert "| --- | --- |" in fragment.content
        assert fragment.table_id == "table_2"
        assert fragment.fragment_count == len(fragments)
        assert fragment.page_numbers == [10, 11]


def test_fragment_indexes_are_sequential() -> None:
    table = TableData(
        table_id="table_3",
        title="Results",
        headers=["A", "B"],
        rows=[
            ["1", "2"],
            ["3", "4"],
            ["5", "6"],
            ["7", "8"],
        ],
        page_numbers=[3],
    )

    fragments = split_table(table, max_tokens=12)

    assert [f.fragment_index for f in fragments] == list(
        range(len(fragments))
    )