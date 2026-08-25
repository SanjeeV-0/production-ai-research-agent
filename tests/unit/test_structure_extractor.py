from app.ingestion.loaders.base import LoadedPage
from app.ingestion.structure import UnitType
from app.ingestion.structure_extractor import StructureExtractor


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
    assert units[2].section_path == (
        "Introduction > Retrieval"
    )


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
            content=(
                "- First item\n"
                "- Second item\n"
                "1. Third item"
            ),
        )
    ]

    units = StructureExtractor().extract(pages)

    assert all(
        unit.unit_type == UnitType.LIST
        for unit in units
    )

    assert len(units) == 3