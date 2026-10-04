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
        """Ask the model to split a multi-intent query into focused queries."""

        response = await self.client.chat.completions.create(
            model=self.model,
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
