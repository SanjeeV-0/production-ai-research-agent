from uuid import uuid4

import pytest

from app.generation.context import ContextAssembler
from app.retrieval.models import RetrievedChunk


def _chunk(
    content: str,
    page_number: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        chunk_id=uuid4(),
        section_id=uuid4(),
        section_path="Results",
        page_numbers=[page_number],
        content=content,
        distance=0.1,
    )


def test_context_assembler_preserves_chunk_order() -> None:
    first = _chunk("First retrieved chunk.", 1)
    second = _chunk("Second retrieved chunk.", 2)

    assembler = ContextAssembler()

    context = assembler.assemble(
        [first, second]
    )

    assert context.text == (
        "[Source 1]\n"
        "First retrieved chunk.\n\n"
        "[Source 2]\n"
        "Second retrieved chunk."
    )

    assert len(context.sources) == 2

    assert context.sources[0].chunk_id == first.chunk_id
    assert context.sources[0].page_numbers == [1]

    assert context.sources[1].chunk_id == second.chunk_id
    assert context.sources[1].page_numbers == [2]


def test_context_assembler_handles_empty_chunks() -> None:
    assembler = ContextAssembler()

    context = assembler.assemble([])

    assert context.text == ""
    assert context.sources == []


def test_context_assembler_respects_character_budget() -> None:
    first = _chunk("First chunk.", 1)
    second = _chunk("Second chunk.", 2)
    third = _chunk("Third chunk.", 3)

    assembler = ContextAssembler(
        max_characters=len(
            "[Source 1]\nFirst chunk.\n\n"
            "[Source 2]\nSecond chunk."
        )
    )

    context = assembler.assemble(
        [first, second, third]
    )

    assert context.text == (
        "[Source 1]\n"
        "First chunk.\n\n"
        "[Source 2]\n"
        "Second chunk."
    )

    assert [
        source.chunk_id
        for source in context.sources
    ] == [
        first.chunk_id,
        second.chunk_id,
    ]


def test_context_assembler_never_partially_includes_chunk() -> None:
    first = _chunk("A" * 20, 1)
    second = _chunk("B" * 100, 2)

    assembler = ContextAssembler(
        max_characters=len("[Source 1]\n" + ("A" * 20))
    )

    context = assembler.assemble(
        [first, second]
    )

    assert context.text == (
        "[Source 1]\n"
        + ("A" * 20)
    )

    assert len(context.sources) == 1
    assert context.sources[0].chunk_id == first.chunk_id


def test_context_assembler_rejects_negative_budget() -> None:
    with pytest.raises(ValueError):
        ContextAssembler(max_characters=-1)