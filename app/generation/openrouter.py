"""The only `GenerationProvider` implementation: calls the answer-generation
LLM through OpenRouter's OpenAI-compatible API.

`temperature`/`max_tokens`/`top_p`/`response_format`/`reasoning_effort` are
explicit, per-instance configuration (sourced from
`Settings.openrouter_generation_*` via `app.core.dependencies.
get_generation_provider`) rather than left to whatever the selected
OpenRouter model defaults to. `max_tokens`/`response_format`/
`reasoning_effort` are optional -- only included in the actual request when
configured (not `None`); `temperature`/`top_p` are always sent explicitly.
No retry/timeout handling is added here.
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
        temperature: float,
        max_tokens: int | None,
        top_p: float,
        response_format: dict[str, object] | None,
        reasoning_effort: str | None,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "Production AI Research & Knowledge Agent",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort

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

        # `temperature`/`top_p` are always explicit, configured values.
        # `max_tokens`/`response_format`/`reasoning_effort` are only
        # included when actually configured (not None), so an unconfigured
        # field never reaches the API as an explicit override.
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
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
        }

        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        if self.response_format is not None:
            request_kwargs["response_format"] = self.response_format

        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort

        response = await self.client.chat.completions.create(**request_kwargs)

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
