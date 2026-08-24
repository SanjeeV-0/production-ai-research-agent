import uuid
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import Document


@pytest.mark.asyncio
async def test_document_persistence() -> None:
    document = Document(
        id=uuid4(),
        title="Test RAG Paper",
        authors="Test Author",
        source="integration-test",
        document_type="research_paper",
        content_hash=f"test-document-hash-{uuid.uuid4()}",
        document_metadata={"topic": "rag"},
    )

    async with async_session_factory() as session:
        session.add(document)
        await session.commit()

        result = await session.execute(
            select(Document).where(Document.id == document.id)
        )
        stored_document = result.scalar_one()

        assert stored_document.title == "Test RAG Paper"
        assert stored_document.document_type == "research_paper"
        assert stored_document.document_metadata == {"topic": "rag"}

        await session.delete(stored_document)
        await session.commit()