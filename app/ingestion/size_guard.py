"""Final step before persistence in the active ingestion pipeline: converts
`SemanticUnit`s into `ChildChunk`s (the shape `ChunkService.persist_chunks`
embeds and stores), enforcing a maximum per-chunk token budget (passed in by
the caller -- hard-coded to 500 in `IngestionService._process_document`).

Also owns `fragment_table_units`, the pre-fragmentation pass that MUST run
on `StructureExtractor`'s output before `shred_semantically`/`apply_size_guard`
ever see a TABLE unit (see `apply_size_guard`'s ValueError below) -- it is
co-located here because both functions enforce the same `max_tokens` budget,
just for different content shapes (prose vs. table rows).

`estimate_tokens` is explicitly a whitespace-word-count approximation, not a
real tokenizer -- documented here as a deliberate, deterministic,
dependency-free placeholder rather than an oversight.
"""

import re
from dataclasses import dataclass
from uuid import UUID

from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import SemanticUnit
from app.ingestion.structure import StructuralUnit, UnitType
from app.ingestion.table_chunker import split_table


@dataclass(frozen=True)
class ChildChunk:
    """Final retrieval unit produced by the size guard."""

    index: int
    content: str
    page_numbers: list[int]
    section_id: UUID | None
    section_path: str | None
    section_level: int
    source_units: tuple[StructuralUnit, ...]


def fragment_table_units(
    units: list[StructuralUnit],
    max_tokens: int,
) -> list[StructuralUnit]:
    """Pre-fragment every TABLE unit so its content already fits `max_tokens`
    before `shred_semantically`/`apply_size_guard` ever see it.

    `shred_semantically` always isolates a TABLE unit into its own
    `SemanticUnit` (never merges it with surrounding prose), and
    `apply_size_guard` cannot itself split an oversized table (splitting a
    markdown table by sentence/word boundaries the way prose is split would
    destroy its row structure) -- it raises instead. So fragmentation MUST
    happen here, upstream, using the existing row-group splitting algorithm
    (`app.ingestion.table_chunker.split_table`), which keeps each fragment a
    valid, self-contained markdown table (headers repeated on every
    fragment) and is called unconditionally for every table, even one
    already under `max_tokens` (it simply returns a single fragment in that
    case).

    Non-TABLE units pass through unchanged. A table whose single largest row
    (with headers) still exceeds `max_tokens` on its own cannot be split
    further by `split_table` (it only splits at row boundaries, not within a
    row) -- that remains a known, documented limitation: such a fragment
    would still trip `apply_size_guard`'s ValueError below.
    """

    fragmented: list[StructuralUnit] = []

    for unit in units:
        if unit.unit_type != UnitType.TABLE or unit.table is None:
            fragmented.append(unit)
            continue

        for fragment in split_table(unit.table, max_tokens=max_tokens):
            fragmented.append(
                StructuralUnit(
                    unit_type=UnitType.TABLE,
                    content=fragment.content,
                    page_numbers=fragment.page_numbers,
                    section_path=unit.section_path,
                    section_level=unit.section_level,
                    section_index=unit.section_index,
                    table=unit.table,
                    metadata={
                        "content_type": "table",
                        "table_id": fragment.table_id,
                        "table_title": fragment.title,
                        "table_fragment_index": fragment.fragment_index,
                        "table_fragment_count": fragment.fragment_count,
                    },
                )
            )

    return fragmented


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using whitespace-separated words.

    This is deterministic and dependency-free. A production tokenizer
    can replace this implementation later.
    """
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentence-like units."""
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )
        if sentence.strip()
    ]


def _split_hard(
    text: str,
    max_tokens: int,
) -> list[str]:
    """Hard-split text when no smaller semantic boundary exists."""
    words = text.split()

    return [
        " ".join(words[index : index + max_tokens])
        for index in range(
            0,
            len(words),
            max_tokens,
        )
    ]


def _split_paragraph(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    Split an oversized paragraph using sentence boundaries first.

    Falls back to hard token splitting for oversized sentences.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]

    sentences = _split_sentences(text)

    if not sentences:
        return _split_hard(text, max_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0

            chunks.extend(
                _split_hard(
                    sentence,
                    max_tokens,
                )
            )
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_prose_unit(
    unit: StructuralUnit,
    max_tokens: int,
) -> list[str]:
    """Split an oversized prose structural unit."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            unit.content.strip(),
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)

        if paragraph_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            chunks.extend(
                _split_paragraph(
                    paragraph,
                    max_tokens,
                )
            )
            continue

        if current and current_tokens + paragraph_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def apply_size_guard(
    semantic_units: list[SemanticUnit],
    section_map: SectionMap,
    max_tokens: int,
) -> list[ChildChunk]:
    """
    Convert semantic units into final child chunks.

    Semantic boundaries are preserved whenever possible.

    Oversized prose is split in this order:

    1. paragraph boundaries
    2. sentence boundaries
    3. hard token boundaries
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    children: list[ChildChunk] = []

    for semantic_unit in semantic_units:
        content = semantic_unit.content

        if not content:
            continue
        section_id = (
            section_map.get(semantic_unit.section_path)
            if semantic_unit.section_path is not None
            else None
        )

        if estimate_tokens(content) <= max_tokens:
            children.append(
                ChildChunk(
                    index=len(children),
                    content=content,
                    page_numbers=semantic_unit.page_numbers,
                    section_id=section_id,
                    section_path=semantic_unit.section_path,
                    section_level=semantic_unit.section_level,
                    source_units=semantic_unit.units,
                )
            )
            continue

        for unit in semantic_unit.units:
            if unit.unit_type == UnitType.TABLE:
                # Reaching here means a TABLE unit arrived without having
                # been pre-fragmented by `fragment_table_units` (or a single
                # fragment's largest row still didn't fit `max_tokens` --
                # see that function's docstring). Tables can't be split by
                # sentence/word boundaries like prose without destroying
                # their structure, so this is a hard stop, not a fallback.
                raise ValueError(
                    "Table exceeds size guard; table fragmentation must occur before size guard."
                )

            pieces = _split_prose_unit(
                unit,
                max_tokens,
            )

            for piece in pieces:
                children.append(
                    ChildChunk(
                        index=len(children),
                        content=piece,
                        page_numbers=list(unit.page_numbers),
                        section_id=section_id,
                        section_path=unit.section_path,
                        section_level=unit.section_level,
                        source_units=(unit,),
                    )
                )

    return children
