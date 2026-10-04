"""The only `GenerationProvider` implementation: calls the answer-generation
LLM through OpenRouter's OpenAI-compatible API.

Deliberately minimal request: no `temperature`, `max_tokens`, `top_p`,
`frequency_penalty`, `presence_penalty`, `response_format`, `timeout`, or
retry parameters are set anywhere in this file -- generation currently runs
entirely on whatever defaults the selected OpenRouter model applies. If
deterministic or length-bounded answers are ever required, those parameters
would need to be added here (and likely exposed via `Settings`, following
the pattern already used for `model`/`base_url`/`app_name`).
"""

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
                    "content": (f"Research context:\n\n{context.text}\n\nQuestion:\n\n{query}"),
                },
            ],
        )

        message = response.choices[0].message.content

        if message is None:
            raise RuntimeError("OpenRouter returned an empty response.")

        usage = response.usage

        return GenerationResult(
            text=message,
            model=response.model or self.model,
            input_tokens=(usage.prompt_tokens if usage is not None else None),
            output_tokens=(usage.completion_tokens if usage is not None else None),
            total_tokens=(usage.total_tokens if usage is not None else None),
        )
