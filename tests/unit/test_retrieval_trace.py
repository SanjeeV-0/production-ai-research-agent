from uuid import uuid4

from app.retrieval.trace import (
    RetrievalTrace,
    RetrievalTraceCandidate,
    RetrievalTraceContext,
)


def test_retrieval_trace_can_capture_generation_context() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    section_id = uuid4()

    source = RetrievalTraceCandidate(
        document_id=document_id,
        chunk_id=chunk_id,
        section_id=section_id,
        section_path="Results",
        page_numbers=[1],
        content="Important research evidence.",
        distance=0.1,
        rerank_score=4.5,
    )

    context = RetrievalTraceContext(
        text="[Source 1]\nImportant research evidence.",
        sources=[source],
    )

    trace = RetrievalTrace(
        query="research query",
        candidate_limit=50,
        candidates=[source],
        final_results=[source],
        context=context,
    )

    assert trace.context is not None
    assert trace.context.text == (
        "[Source 1]\nImportant research evidence."
    )

    assert len(trace.context.sources) == 1

    assert (
        trace.context.sources[0].chunk_id
        == chunk_id
    )