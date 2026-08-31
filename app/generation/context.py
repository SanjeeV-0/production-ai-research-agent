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

    def __init__(self, max_characters: int | None = None) -> None:
        if max_characters is not None and max_characters < 0:
            raise ValueError(
                "max_characters must be greater than or equal to zero."
            )

        self.max_characters = max_characters

    def assemble(
        self,
        chunks: list[RetrievedChunk],
    ) -> GenerationContext:
        """Build generation context from retrieved chunks."""

        text_parts: list[str] = []
        sources: list[GenerationContextSource] = []
        current_length = 0

        for index, chunk in enumerate(chunks, start=1):
            source_text = f"[Source {index}]\n{chunk.content}"

            additional_length = len(source_text)

            if text_parts:
                additional_length += 2

            if (
                self.max_characters is not None
                and current_length + additional_length
                > self.max_characters
            ):
                break

            text_parts.append(source_text)

            sources.append(
                GenerationContextSource(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    section_id=chunk.section_id,
                    section_path=chunk.section_path,
                    page_numbers=chunk.page_numbers,
                )
            )

            current_length += additional_length

        return GenerationContext(
            text="\n\n".join(text_parts),
            sources=sources,
        )