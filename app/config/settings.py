"""Single source of truth for all runtime configuration.

Every tunable value that isn't a hard-coded algorithm constant (chunking
thresholds, decomposition temperature, etc. -- those live next to the code
that uses them) lives here. Values are read from the environment / `.env`
(case-insensitive) with the defaults below applied when unset. Note that
`.env.example` only documents a subset of these fields -- every field here
still works via its environment variable even if absent from that example
file.

`OPENROUTER_MODEL` is shared by both query decomposition
(`app.retrieval.openrouter_decomposition`) and answer generation
(`app.generation.openrouter`) -- there is intentionally no separate
decomposition-model setting.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production AI Research & Knowledge Agent"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/research_agent"
    # Root directory for LocalFileStorage. Changing this does not move
    # already-stored files -- it only changes where new ones are written.
    storage_root: str = "data"
    # Output dimension is fixed in the schema (Vector(384) on
    # DocumentChunk.embedding). Swapping this for a model with a different
    # dimension requires a migration AND re-embedding every existing chunk;
    # even a same-dimension swap requires re-embedding, since vectors from
    # different models are not comparable.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Server-side capability ceiling for retrieval trace capture. A client's
    # per-request `trace=true` can never force a trace when this is false --
    # see app.api.retrieval / app.api.research's `effective_trace` logic.
    # Never gates retrieval or query decomposition themselves.
    trace_enabled: bool = False

    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    # Required by BOTH the query-decomposition provider and the generation
    # provider (app.core.dependencies raises eagerly if this is unset) --
    # in practice this makes decomposition mandatory, not optional, since
    # the DI wiring always constructs a decomposer for the retrieval service.
    openrouter_api_key: str | None = None
    # Shared by query decomposition AND answer generation -- there is no
    # separate decomposition-model setting.
    openrouter_model: str = "meta-llama/llama-3.3-8b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_name: str = "Production AI Research & Knowledge Agent"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
