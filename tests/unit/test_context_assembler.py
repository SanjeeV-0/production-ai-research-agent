from uuid import uuid4

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