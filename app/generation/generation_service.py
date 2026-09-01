from app.generation.context import GenerationContext
from app.generation.service import (
    GenerationProvider,
    GenerationResult,
)
from app.observability.langfuse import get_langfuse


class GenerationService:
    """Coordinates context and LLM generation."""

    def __init__(
        self,
        provider: GenerationProvider,
    ) -> None:
        self.provider = provider

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using the supplied context."""

        langfuse = get_langfuse()

        if langfuse is None:
            return await self.provider.generate(
                query=query,
                context=context,
            )

        with langfuse.start_as_current_observation(
            as_type="generation",
            name="answer-generation",
            input={
                "query": query,
                "context": context.text,
            },
        ) as observation:
            try:
                result = await self.provider.generate(
                    query=query,
                    context=context,
                )

                observation.update(
                    output={
                        "text": result.text,
                        "model": result.model,
                    },
                )

                return result

            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=str(exc),
                )
                raise