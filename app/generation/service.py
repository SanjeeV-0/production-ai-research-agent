from typing import Protocol

from app.generation.context import GenerationContext


class GenerationResult:
    """Result returned by a generation provider."""

    def __init__(
        self,
        text: str,
        model: str,
    ) -> None:
        self.text = text
        self.model = model


class GenerationProvider(Protocol):
    """Interface implemented by an LLM generation provider."""

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using the supplied context."""