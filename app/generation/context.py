from dataclasses import dataclass
from uuid import UUID

from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True)
class GenerationContextSource:
    """Source metadata for a chunk used during generation."""

    document_id: UUID
    chunk_id: UUID
    section_id: UUID
    section_path: str
    page_numbers: list[int]


@dataclass(frozen=True)
class GenerationContext:
    """Context assembled for a future generation request."""

    text: str
    sources: list[GenerationContextSource]


class ContextAssembler:
    """Assemble retrieved chunks into generation context."""

    def assemble(
        self,
        chunks: list[RetrievedChunk],
    ) -> GenerationContext:
        """Build generation context from retrieved chunks."""

        text_parts: list[str] = []
        sources: list[GenerationContextSource] = []

        for index, chunk in enumerate(chunks, start=1):
            text_parts.append(
                f"[Source {index}]\n{chunk.content}"
            )

            sources.append(
                GenerationContextSource(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    section_id=chunk.section_id,
                    section_path=chunk.section_path,
                    page_numbers=chunk.page_numbers,
                )
            )

        return GenerationContext(
            text="\n\n".join(text_parts),
            sources=sources,
        )