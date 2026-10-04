from app.ingestion.loaders.base import LoadedPage
from app.ingestion.structure import UnitType
from app.ingestion.structure_extractor import StructureExtractor
from app.ingestion.table_serializer import table_to_markdown


def test_extracts_markdown_headings_and_paragraphs() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "# Introduction\n\n"
                "This is the introduction.\n\n"
                "## Retrieval\n\n"
                "Retrieval improves search."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.PARAGRAPH,
        UnitType.HEADING,
        UnitType.PARAGRAPH,
    ]

    assert units[0].section_path == "Introduction"
    assert units[2].section_path == ("Introduction > Retrieval")


def test_preserves_page_numbers() -> None:
    pages = [
        LoadedPage(
            page_number=3,
            content="Some content.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert len(units) == 1
    assert units[0].page_numbers == [3]


def test_extracts_lists() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=("- First item\n- Second item\n1. Third item"),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert all(unit.unit_type == UnitType.LIST for unit in units)

    assert len(units) == 3


def test_extracts_markdown_table_under_a_heading() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=("# Results\n\n| Model | Score |\n| --- | --- |\n| A | 0.90 |\n| B | 0.85 |\n"),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.HEADING,
        UnitType.TABLE,
    ]

    table_unit = units[1]
    assert table_unit.section_path == "Results"
    assert table_unit.page_numbers == [1]
    assert table_unit.table is not None
    assert table_unit.table.headers == ["Model", "Score"]
    assert table_unit.table.rows == [["A", "0.90"], ["B", "0.85"]]

    # content is re-serialized through the canonical table_serializer, not
    # kept as the raw source lines verbatim.
    assert table_unit.content == table_to_markdown(table_unit.table)


def test_extracts_text_and_table_together() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "Intro paragraph before the table.\n\n"
                "| A | B |\n"
                "| --- | --- |\n"
                "| 1 | 2 |\n\n"
                "Paragraph after the table."
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert [unit.unit_type for unit in units] == [
        UnitType.PARAGRAPH,
        UnitType.TABLE,
        UnitType.PARAGRAPH,
    ]

    assert units[0].content == "Intro paragraph before the table."
    assert units[2].content == "Paragraph after the table."


def test_extracts_multiple_tables_with_distinct_table_ids() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content=(
                "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |\n"
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    table_units = [unit for unit in units if unit.unit_type == UnitType.TABLE]

    assert len(table_units) == 2
    assert table_units[0].table.table_id != table_units[1].table.table_id
    assert table_units[0].table.headers == ["A", "B"]
    assert table_units[1].table.headers == ["C", "D"]


def test_headingless_table_has_empty_section_path() -> None:
    """A table with no preceding heading must not be assigned a fabricated
    section -- it gets the same empty section_path as any other headingless
    unit, which downstream (SemanticUnit/apply_size_guard) becomes `None`."""

    pages = [
        LoadedPage(
            page_number=1,
            content=("| A | B |\n| --- | --- |\n| 1 | 2 |\n"),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert len(units) == 1
    assert units[0].unit_type == UnitType.TABLE
    assert units[0].section_path == ""


def test_table_row_without_separator_is_not_detected_as_a_table() -> None:
    """A line that merely contains pipe characters is not a table unless
    immediately followed by a valid GFM separator row -- otherwise it (and
    the ordinary text line after it) is just paragraph text."""

    pages = [
        LoadedPage(
            page_number=1,
            content="Cost | Benefit analysis is not a table.\nIt is followed by plain text.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert len(units) == 1
    assert units[0].unit_type == UnitType.PARAGRAPH


def test_horizontal_rule_is_not_mistaken_for_a_table_separator() -> None:
    pages = [
        LoadedPage(
            page_number=1,
            content="Some paragraph line.\n\n---\n\nAnother paragraph.",
        )
    ]

    units = StructureExtractor().extract(pages)

    assert all(unit.unit_type == UnitType.PARAGRAPH for unit in units)
