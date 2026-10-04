"""`group_structural_units` itself is NOT part of the active ingestion
pipeline -- `IngestionService` (app.ingestion.service) uses
`app.ingestion.semantic_shredder` + `app.ingestion.size_guard` instead, and
never calls this function. Only exercised by its own unit/integration tests
(tests/unit/test_structure_chunker.py,
tests/integration/test_table_chunk_parent_context.py).

Unlike when this note was first written, the table-handling modules this
file depends on (`app.ingestion.table_chunker.split_table`,
`app.ingestion.table_serializer.table_to_markdown`) ARE now part of the
active pipeline -- they're called directly from
`app.ingestion.size_guard.fragment_table_units` and
`app.ingestion.structure_extractor.StructureExtractor`. It is specifically
the candidate-grouping strategy in THIS file (as an alternative to
semantic_shredder/size_guard) that remains unused, not table support itself.
"""

from dataclasses import dataclass

from app.ingestion.structure import StructuralUnit, UnitType
from app.ingestion.table_chunker import split_table


@dataclass(frozen=True)
class CandidateGroup:
    """A deterministic retrieval candidate."""

    content: str
    page_numbers: list[int]
    section_path: str
    section_level: int
    unit_types: list[UnitType]
    metadata: dict


def _create_group(
    units: list[StructuralUnit],
) -> CandidateGroup:
    """Create a candidate from structural units."""
    page_numbers = sorted({page for unit in units for page in unit.page_numbers})

    return CandidateGroup(
        content="\n\n".join(unit.content for unit in units if unit.content),
        page_numbers=page_numbers,
        section_path=units[0].section_path,
        section_level=units[0].section_level,
        unit_types=[unit.unit_type for unit in units],
        metadata={},
    )


def group_structural_units(
    units: list[StructuralUnit],
    table_max_tokens: int = 600,
) -> list[CandidateGroup]:
    """Convert structural units into deterministic candidates."""
    groups: list[CandidateGroup] = []
    current_units: list[StructuralUnit] = []

    def flush() -> None:
        if current_units:
            groups.append(_create_group(current_units))
            current_units.clear()

    for unit in units:
        if unit.unit_type == UnitType.HEADING:
            flush()
            current_units.append(unit)
            continue

        if unit.unit_type == UnitType.TABLE:
            flush()

            if unit.table is None:
                continue

            fragments = split_table(
                unit.table,
                max_tokens=table_max_tokens,
            )

            for fragment in fragments:
                groups.append(
                    CandidateGroup(
                        content=fragment.content,
                        page_numbers=fragment.page_numbers,
                        section_path=unit.section_path,
                        section_level=unit.section_level,
                        unit_types=[UnitType.TABLE],
                        metadata={
                            "content_type": "table",
                            "table_id": fragment.table_id,
                            "table_title": fragment.title,
                            "table_fragment_index": (fragment.fragment_index),
                            "table_fragment_count": (fragment.fragment_count),
                        },
                    )
                )

            continue

        if current_units and unit.section_path != current_units[0].section_path:
            flush()

        current_units.append(unit)

    flush()

    return groups
