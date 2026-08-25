import re
from dataclasses import dataclass
from uuid import UUID

from app.ingestion.section_map import SectionMap
from app.ingestion.semantic_shredder import SemanticUnit
from app.ingestion.structure import StructuralUnit, UnitType


@dataclass(frozen=True)
class ChildChunk:
    """Final retrieval unit produced by the size guard."""

    index: int
    content: str
    page_numbers: list[int]
    section_id: UUID
    section_path: str
    section_level: int
    source_units: tuple[StructuralUnit, ...]


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
        " ".join(words[index:index + max_tokens])
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

        if (
            current
            and current_tokens + sentence_tokens > max_tokens
        ):
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

        if (
            current
            and current_tokens + paragraph_tokens > max_tokens
        ):
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
        raise ValueError(
            "max_tokens must be greater than zero"
        )
        
    children: list[ChildChunk] = []

    for semantic_unit in semantic_units:
        content = semantic_unit.content

        if not content:
            continue
        section_id = section_map.get(
            semantic_unit.section_path
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
                raise ValueError(
                    "Table exceeds size guard; table fragmentation "
                    "must occur before size guard."
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