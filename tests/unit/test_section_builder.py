from app.ingestion.section_builder import SectionBuilder
from app.ingestion.structure import (
    StructuralUnit,
    UnitType,
)


def _unit(
    section_path: str,
    section_level: int,
    section_index: int,
) -> StructuralUnit:
    return StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content="content",
        page_numbers=[1],
        section_path=section_path,
        section_level=section_level,
        section_index=section_index,
    )


def test_builds_top_level_sections_in_document_order() -> None:
    units = [
        _unit("Introduction", 1, 0),
        _unit("Introduction", 1, 1),
        _unit("Methodology", 1, 0),
        _unit("Results", 1, 0),
    ]

    result = SectionBuilder().build(units)

    assert [node.path for node in result] == [
        "Introduction",
        "Methodology",
        "Results",
    ]

    assert [node.index for node in result] == [
        0,
        1,
        2,
    ]


def test_builds_nested_sections_with_parent_paths() -> None:
    units = [
        _unit("Results", 1, 0),
        _unit("Results > Retrieval", 2, 0),
        _unit("Results > Evaluation", 2, 1),
    ]

    result = SectionBuilder().build(units)

    assert [
        (
            node.path,
            node.parent_path,
            node.level,
            node.index,
        )
        for node in result
    ] == [
        ("Results", None, 1, 0),
        ("Results > Retrieval", "Results", 2, 0),
        ("Results > Evaluation", "Results", 2, 1),
    ]


def test_duplicate_section_paths_create_one_node() -> None:
    units = [
        _unit("Results", 1, 0),
        _unit("Results > Retrieval", 2, 0),
        _unit("Results > Retrieval", 2, 1),
        _unit("Results > Retrieval", 2, 2),
    ]

    result = SectionBuilder().build(units)

    assert [node.path for node in result] == [
        "Results",
        "Results > Retrieval",
    ]


def test_nested_section_without_explicit_parent_is_completed() -> None:
    units = [
        _unit("Results > Retrieval", 2, 0),
    ]

    result = SectionBuilder().build(units)

    assert [node.path for node in result] == [
        "Results",
        "Results > Retrieval",
    ]

    assert result[0].parent_path is None
    assert result[1].parent_path == "Results"


def test_structural_unit_index_does_not_determine_section_index() -> None:
    units = [
        _unit("Introduction", 1, 99),
        _unit("Methodology", 1, 42),
        _unit("Results", 1, 7),
    ]

    result = SectionBuilder().build(units)

    assert [node.index for node in result] == [
        0,
        1,
        2,
    ]