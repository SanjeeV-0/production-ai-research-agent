from app.ingestion.structure import (
    StructuralUnit,
    TableData,
    UnitType,
)
from app.ingestion.structure_chunker import group_structural_units


def test_large_table_creates_children_under_same_section() -> None:
    table = TableData(
        table_id="table_1",
        title="Retrieval Results",
        headers=["Model", "Recall"],
        rows=[
            ["A", "0.80"],
            ["B", "0.82"],
            ["C", "0.85"],
            ["D", "0.88"],
            ["E", "0.91"],
        ],
        page_numbers=[10, 11],
    )

    units = [
        StructuralUnit(
            unit_type=UnitType.PARAGRAPH,
            content="The results show improved retrieval.",
            page_numbers=[10],
            section_path="3 Results",
            section_level=1,
            section_index=0,
        ),
        StructuralUnit(
            unit_type=UnitType.TABLE,
            content="",
            page_numbers=[10, 11],
            section_path="3 Results",
            section_level=1,
            section_index=1,
            table=table,
        ),
        StructuralUnit(
            unit_type=UnitType.PARAGRAPH,
            content="The table demonstrates the improvement.",
            page_numbers=[11],
            section_path="3 Results",
            section_level=1,
            section_index=2,
        ),
    ]

    groups = group_structural_units(
    units,
    table_max_tokens=15,
    )

    table_groups = [
        group
        for group in groups
        if group.metadata.get("content_type") == "table"
    ]

    assert len(table_groups) > 1

    for group in table_groups:
        assert group.section_path == "3 Results"
        assert group.unit_types == [UnitType.TABLE]
        assert group.metadata["table_id"] == "table_1"