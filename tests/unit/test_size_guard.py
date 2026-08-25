from app.ingestion.semantic_shredder import SemanticUnit
from app.ingestion.size_guard import apply_size_guard
from app.ingestion.structure import StructuralUnit, UnitType


def _paragraph(
    content: str,
    section_path: str = "Results",
    index: int = 0,
) -> StructuralUnit:
    return StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content=content,
        page_numbers=[1],
        section_path=section_path,
        section_level=1,
        section_index=index,
    )


def test_small_semantic_unit_remains_one_child() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "Retrieval improves document search.",
            ),
        )
    )

    result = apply_size_guard(
        [semantic_unit],
        max_tokens=10,
    )

    assert len(result) == 1
    assert result[0].content == (
        "Retrieval improves document search."
    )


def test_oversized_semantic_unit_is_split() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three four five six seven eight nine ten",
            ),
        )
    )

    result = apply_size_guard(
        [semantic_unit],
        max_tokens=5,
    )

    assert len(result) == 2
    assert all(
        len(child.content.split()) <= 5
        for child in result
    )


def test_semantic_unit_preserves_section() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three",
                section_path="Results",
            ),
        )
    )

    result = apply_size_guard(
        [semantic_unit],
        max_tokens=2,
    )

    assert all(
        child.section_path == "Results"
        for child in result
    )


def test_page_provenance_is_preserved() -> None:
    unit = StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content="one two three four",
        page_numbers=[3, 4],
        section_path="Results",
        section_level=1,
        section_index=0,
    )

    result = apply_size_guard(
        [SemanticUnit((unit,))],
        max_tokens=2,
    )

    assert all(
        child.page_numbers == [3, 4]
        for child in result
    )


def test_invalid_max_tokens_raises() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph("some content"),
        )
    )

    try:
        apply_size_guard(
            [semantic_unit],
            max_tokens=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid max_tokens"
        )

def test_oversized_paragraph_prefers_sentence_boundaries() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "One two three. "
                "Four five six. "
                "Seven eight nine.",
            ),
        )
    )

    result = apply_size_guard(
        [semantic_unit],
        max_tokens=6,
    )

    assert len(result) == 2
    assert result[0].content == "One two three. Four five six."
    assert result[1].content == "Seven eight nine."


def test_oversized_sentence_uses_hard_boundary() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three four five six seven eight nine ten",
            ),
        )
    )

    result = apply_size_guard(
        [semantic_unit],
        max_tokens=5,
    )

    assert len(result) == 2
    assert result[0].content == "one two three four five"
    assert result[1].content == "six seven eight nine ten"


def test_multiple_paragraphs_prefer_paragraph_boundary() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "One two three.\n\nFour five six.",
            ),
        )
    )

    result = apply_size_guard(
        [semantic_unit],
        max_tokens=3,
    )

    assert len(result) == 2
    assert result[0].content == "One two three."
    assert result[1].content == "Four five six."