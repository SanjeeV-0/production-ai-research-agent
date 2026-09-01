from openai import AsyncOpenAI

from app.generation.context import GenerationContext
from app.generation.service import GenerationResult


class OpenRouterGenerationProvider:
    """Generate answers through the OpenRouter OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "Production AI Research & Knowledge Agent",
    ) -> None:
        self.model = model

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "X-Title": app_name,
            },
        )

    async def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate an answer using OpenRouter."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Answer the user's question using the "
                        "provided research context. "
                        "Do not invent facts that are not supported "
                        "by the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research context:\n\n"
                        f"{context.text}\n\n"
                        f"Question:\n\n"
                        f"{query}"
                    ),
                },
            ],
        )

        message = response.choices[0].message.content

        if message is None:
            raise RuntimeError(
                "OpenRouter returned an empty response."
            )

        usage = response.usage

        return GenerationResult(
            text=message,
            model=response.model or self.model,
            input_tokens=(
                usage.prompt_tokens
                if usage is not None
                else None
            ),
            output_tokens=(
                usage.completion_tokens
                if usage is not None
                else None
            ),
            total_tokens=(
                usage.total_tokens
                if usage is not None
                else None
            ),
        )