from collections.abc import Sequence
from dataclasses import dataclass

from app.embeddings.provider import EmbeddingProvider
from app.ingestion.similarity import cosine_similarity
from app.ingestion.structure import StructuralUnit, UnitType


@dataclass(frozen=True)
class SemanticUnit:
    """A semantically coherent sequence of structural units."""

    units: tuple[StructuralUnit, ...]

    @property
    def content(self) -> str:
        """Return the combined textual content."""
        return "\n\n".join(
            unit.content
            for unit in self.units
            if unit.content
        )

    @property
    def page_numbers(self) -> list[int]:
        """Return all source pages represented by the unit."""
        return sorted(
            {
                page
                for unit in self.units
                for page in unit.page_numbers
            }
        )

    @property
    def section_path(self) -> str:
        """Return the owning section path."""
        return self.units[0].section_path

    @property
    def section_level(self) -> int:
        """Return the owning section level."""
        return self.units[0].section_level


def _is_prose(unit: StructuralUnit) -> bool:
    """Return whether a structural unit can participate in shredding."""
    return unit.unit_type in {
        UnitType.PARAGRAPH,
        UnitType.LIST,
    }


def _prose_units(
    units: Sequence[StructuralUnit],
) -> list[StructuralUnit]:
    """Extract prose units that require semantic embeddings."""
    return [
        unit
        for unit in units
        if _is_prose(unit)
    ]


def shred_semantically(
    units: Sequence[StructuralUnit],
    embedding_provider: EmbeddingProvider,
    threshold: float,
) -> list[SemanticUnit]:
    """
    Split ordered structural units using adjacent semantic boundaries.

    Prose units are embedded once in a batch. Semantic similarity is
    calculated only between adjacent prose units. No global clustering
    is performed.
    """
    semantic_units: list[SemanticUnit] = []
    current: list[StructuralUnit] = []

    prose = _prose_units(units)
    embeddings = embedding_provider.embed_batch(
        [unit.content for unit in prose]
    )

    embedding_by_unit = {
        id(unit): embedding
        for unit, embedding in zip(
            prose,
            embeddings,
            strict=True,
        )
    }

    def flush() -> None:
        if current:
            semantic_units.append(
                SemanticUnit(tuple(current))
            )
            current.clear()

    previous_prose: StructuralUnit | None = None

    for unit in units:
        if unit.unit_type == UnitType.HEADING:
            flush()
            previous_prose = None
            continue

        if unit.unit_type == UnitType.TABLE:
            flush()

            semantic_units.append(
                SemanticUnit((unit,))
            )

            previous_prose = None
            continue

        if not _is_prose(unit):
            flush()
            previous_prose = None

            semantic_units.append(
                SemanticUnit((unit,))
            )
            continue

        if not current:
            current.append(unit)
            previous_prose = unit
            continue

        if (
            previous_prose is None
            or unit.section_path != previous_prose.section_path
        ):
            flush()
            current.append(unit)
            previous_prose = unit
            continue

        previous_embedding = embedding_by_unit[id(previous_prose)]
        current_embedding = embedding_by_unit[id(unit)]

        score = cosine_similarity(
            previous_embedding,
            current_embedding,
        )

        if score >= threshold:
            current.append(unit)
        else:
            flush()
            current.append(unit)

        previous_prose = unit

    flush()

    return semantic_units