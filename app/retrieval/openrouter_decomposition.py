"""The only `QueryDecompositionProvider` implementation: calls an LLM
through OpenRouter's OpenAI-compatible API to split a query into focused
retrieval sub-queries. Raw/malformed output is this provider's problem to
raise on (ValueError/RuntimeError) -- validating and recovering from that is
`app.retrieval.query_decomposer.QueryDecomposer`'s job, not this file's.
"""

import json

from openai import AsyncOpenAI


class OpenRouterQueryDecompositionProvider:
    """Generate focused retrieval queries through OpenRouter."""

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

    async def decompose(self, query: str) -> list[str]:
        """Ask the model to split a multi-intent query into focused queries.

        Raises `RuntimeError`/`ValueError` for an empty response or any
        shape that doesn't match `{"queries": [str, ...]}` -- these are
        caught by `QueryDecomposer.decompose`, which falls back to the
        original query alone rather than propagating the failure.
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            # Hard-coded (not a Settings field): decomposition should be
            # deterministic given the same input, since it's a structural
            # query-planning step, not a creative one.
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval query planner. "
                        "Split the user's question only when it contains "
                        "multiple distinct information needs. "
                        "Generate focused retrieval questions for those needs. "
                        "Do not generate paraphrases, semantic variants, "
                        "or arbitrary query expansions. "
                        "Return at most five queries. "
                        'Return JSON only in the form {"queries": ["..."]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        message = response.choices[0].message.content

        if message is None:
            raise RuntimeError("OpenRouter returned an empty decomposition response.")

        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenRouter returned invalid JSON for query decomposition.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Query decomposition response must be a JSON object.")

        queries = payload.get("queries")

        if not isinstance(queries, list):
            raise ValueError("Query decomposition response must contain a 'queries' list.")

        if not all(isinstance(query, str) for query in queries):
            raise ValueError("All decomposed queries must be strings.")

        return queries
