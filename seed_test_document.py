# import asyncio
# import selectors
# from pathlib import Path

# from app.core.database import async_session_factory
# from app.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddingProvider,
# )
# from app.ingestion.loaders.markdown import MarkdownLoader
# from app.ingestion.service import IngestionService


# DOCUMENT_PATH = Path("test_research.md")


# DOCUMENT_PATH.write_text(
#     """# Retrieval Augmented Generation

# Retrieval augmented generation combines information retrieval
# with language generation.

# ## Retrieval

# Vector embeddings allow documents to be represented as numerical
# vectors. Similar vectors can be retrieved using cosine similarity.

# ## Generation

# The retrieved context can then be provided to a language model
# to improve the relevance and grounding of generated answers.
# """,
#     encoding="utf-8",
# )


# async def main() -> None:
#     embedding_provider = SentenceTransformerEmbeddingProvider()

#     async with async_session_factory() as session:
#         service = IngestionService(
#             session,
#             embedding_provider=embedding_provider,
#         )

#         document = await service.ingest_file(
#             path=DOCUMENT_PATH,
#             loader=MarkdownLoader(),
#             title="RAG Retrieval Test",
#             document_type="research_paper",
#             source="local-test",
#         )

#         await session.commit()

#         print(f"Document ID: {document.id}")


# if __name__ == "__main__":
#     asyncio.run(
#         main(),
#         loop_factory=lambda: asyncio.SelectorEventLoop(
#             selectors.SelectSelector()
#         ),
#     )

#     asyncio.run(main())

import asyncio
import selectors
from pathlib import Path

from app.core.database import async_session_factory
from app.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.service import IngestionService

DOCUMENT_PATH = Path("test_research.md")


DOCUMENT_PATH.write_text(
    """# Retrieval Augmented Generation

Retrieval augmented generation combines information retrieval
with language generation.

## Retrieval

Vector embeddings allow documents to be represented as numerical
vectors. Similar vectors can be retrieved using cosine similarity.

## Generation

The retrieved context can then be provided to a language model
to improve the relevance and grounding of generated answers.
""",
    encoding="utf-8",
)


async def main() -> None:
    embedding_provider = SentenceTransformerEmbeddingProvider()

    async with async_session_factory() as session:
        service = IngestionService(
            session,
            embedding_provider=embedding_provider,
        )

        document = await service.ingest_file(
            path=DOCUMENT_PATH,
            loader=MarkdownLoader(),
            title="RAG Retrieval Test",
            document_type="research_paper",
            source="local-test",
        )

        await session.commit()

        print(f"Document ID: {document.id}")


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        ),
    )
