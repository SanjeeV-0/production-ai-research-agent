from uuid import UUID, uuid4

from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import SemanticUnit
from app.ingestion.size_guard import apply_size_guard, fragment_table_units
from app.ingestion.structure import StructuralUnit, TableData, UnitType


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


def _section_map(
    section_path: str = "Results",
) -> tuple[SectionMap, UUID]:
    section_map = SectionMap()
    section_id = uuid4()

    section_map.add(
        section_path,
        section_id,
    )

    return section_map, section_id


def test_small_semantic_unit_remains_one_child() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "Retrieval improves document search.",
            ),
        )
    )

    section_map, section_id = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=10,
    )

    assert len(result) == 1
    assert result[0].content == ("Retrieval improves document search.")
    assert result[0].section_id == section_id


def test_oversized_semantic_unit_is_split() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three four five six seven eight nine ten",
            ),
        )
    )

    section_map, section_id = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=5,
    )

    assert len(result) == 2

    assert all(len(child.content.split()) <= 5 for child in result)

    assert all(child.section_id == section_id for child in result)


def test_semantic_unit_preserves_section() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three",
                section_path="Results",
            ),
        )
    )

    section_map, section_id = _section_map("Results")

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=2,
    )

    assert all(child.section_path == "Results" for child in result)

    assert all(child.section_id == section_id for child in result)


def test_page_provenance_is_preserved() -> None:
    unit = StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content="one two three four",
        page_numbers=[3, 4],
        section_path="Results",
        section_level=1,
        section_index=0,
    )

    section_map, _ = _section_map()

    result = apply_size_guard(
        [SemanticUnit((unit,))],
        section_map=section_map,
        max_tokens=2,
    )

    assert all(child.page_numbers == [3, 4] for child in result)


def test_invalid_max_tokens_raises() -> None:
    semantic_unit = SemanticUnit((_paragraph("some content"),))

    section_map, _ = _section_map()

    try:
        apply_size_guard(
            [semantic_unit],
            section_map=section_map,
            max_tokens=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid max_tokens")


def test_oversized_paragraph_prefers_sentence_boundaries() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "One two three. Four five six. Seven eight nine.",
            ),
        )
    )

    section_map, _ = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=6,
    )

    assert len(result) == 2
    assert result[0].content == ("One two three. Four five six.")
    assert result[1].content == ("Seven eight nine.")


def test_oversized_sentence_uses_hard_boundary() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "one two three four five six seven eight nine ten",
            ),
        )
    )

    section_map, _ = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=5,
    )

    assert len(result) == 2
    assert result[0].content == ("one two three four five")
    assert result[1].content == ("six seven eight nine ten")


def _table_unit(
    rows: list[list[str]],
    section_path: str = "Results",
) -> StructuralUnit:
    table = TableData(
        table_id="table-1",
        title="Retrieval Results",
        headers=["Model", "Score"],
        rows=rows,
        page_numbers=[10],
    )

    return StructuralUnit(
        unit_type=UnitType.TABLE,
        content="placeholder -- fragment_table_units always re-serializes",
        page_numbers=[10],
        section_path=section_path,
        section_level=1,
        section_index=0,
        table=table,
    )


def test_fragment_table_units_leaves_non_table_units_unchanged() -> None:
    paragraph = _paragraph("Ordinary prose.")

    result = fragment_table_units([paragraph], max_tokens=500)

    assert result == [paragraph]


def test_fragment_table_units_small_table_becomes_one_fragment() -> None:
    unit = _table_unit(rows=[["A", "0.90"], ["B", "0.85"]])

    result = fragment_table_units([unit], max_tokens=500)

    assert len(result) == 1
    assert result[0].unit_type == UnitType.TABLE
    assert "| Model | Score |" in result[0].content
    assert "| A | 0.90 |" in result[0].content
    assert result[0].metadata["content_type"] == "table"
    assert result[0].metadata["table_id"] == "table-1"
    assert result[0].metadata["table_fragment_index"] == 0
    assert result[0].metadata["table_fragment_count"] == 1
    # Provenance carried over unchanged from the source unit.
    assert result[0].section_path == "Results"
    assert result[0].page_numbers == [10]


def test_fragment_table_units_large_table_produces_multiple_fragments() -> None:
    rows = [[f"Model{i}", f"0.{90 - i}"] for i in range(10)]
    unit = _table_unit(rows=rows)

    result = fragment_table_units([unit], max_tokens=20)

    assert len(result) > 1
    assert all(u.unit_type == UnitType.TABLE for u in result)

    fragment_indexes = [u.metadata["table_fragment_index"] for u in result]
    assert fragment_indexes == list(range(len(result)))

    for fragment in result:
        assert fragment.metadata["table_id"] == "table-1"
        assert fragment.metadata["table_fragment_count"] == len(result)
        # Every fragment repeats the header so each is independently
        # understandable without the others.
        assert "| Model | Score |" in fragment.content


def test_fragment_table_units_preserves_each_fragment_within_apply_size_guard() -> None:
    """End-to-end: a table pre-fragmented by fragment_table_units must never
    trigger apply_size_guard's "table exceeds size guard" safety net, since
    every fragment it produces already fits the same max_tokens budget."""

    rows = [[f"Model{i}", f"0.{90 - i}"] for i in range(10)]
    unit = _table_unit(rows=rows)

    fragments = fragment_table_units([unit], max_tokens=20)

    section_map, section_id = _section_map("Results")

    result = apply_size_guard(
        [SemanticUnit((fragment,)) for fragment in fragments],
        section_map=section_map,
        max_tokens=20,
    )

    assert len(result) == len(fragments)
    assert all(child.section_id == section_id for child in result)


def test_single_row_fragment_still_exceeding_budget_trips_the_safety_net() -> None:
    """Documented, known limitation: split_table only splits at row
    boundaries, never within a row. If a single row (plus headers/title)
    already exceeds max_tokens on its own, fragment_table_units cannot
    shrink it further, and apply_size_guard's ValueError correctly fires as
    the last-resort safety net rather than silently emitting an oversized
    chunk."""

    unit = _table_unit(rows=[["A", "0.90"]])

    # max_tokens=5 is smaller than even one header+separator+row combo,
    # which fragment_table_units (via table_chunker.split_table) cannot
    # shrink below one row.
    fragments = fragment_table_units([unit], max_tokens=5)

    section_map, _ = _section_map("Results")

    try:
        apply_size_guard(
            [SemanticUnit((fragment,)) for fragment in fragments],
            section_map=section_map,
            max_tokens=5,
        )
    except ValueError as exc:
        assert "Table exceeds size guard" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unshrinkable oversized table row")


def test_multiple_paragraphs_prefer_paragraph_boundary() -> None:
    semantic_unit = SemanticUnit(
        (
            _paragraph(
                "One two three.\n\nFour five six.",
            ),
        )
    )

    section_map, _ = _section_map()

    result = apply_size_guard(
        [semantic_unit],
        section_map=section_map,
        max_tokens=3,
    )

    assert len(result) == 2
    assert result[0].content == ("One two three.")
    assert result[1].content == ("Four five six.")
