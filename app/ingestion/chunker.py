from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text with its source page numbers."""

    index: int
    content: str
    page_numbers: list[int]


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Split page content into overlapping chunks while preserving pages."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be between zero and chunk_size - 1"
        )

    chunks: list[TextChunk] = []
    current_text = ""
    current_pages: list[int] = []

    for page_number, content in pages:
        words = content.split()

        for word in words:
            current_text = f"{current_text} {word}".strip()

            if page_number not in current_pages:
                current_pages.append(page_number)

            if len(current_text) >= chunk_size:
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        content=current_text,
                        page_numbers=current_pages.copy(),
                    )
                )

                overlap_text = current_text[-chunk_overlap:]
                current_text = overlap_text

                current_pages = [page_number]

    if current_text:
        chunks.append(
            TextChunk(
                index=len(chunks),
                content=current_text,
                page_numbers=current_pages.copy(),
            )
        )

    return chunks