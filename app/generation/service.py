from dataclasses import dataclass
from typing import Protocol

from app.generation.context import GenerationContext


@dataclass(frozen=True)
class GenerationResult:
    """Result returned by a generation provider."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GenerationProvider(Protocol):
    """Interface implemented by an LLM generation provider."""

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using the supplied context."""