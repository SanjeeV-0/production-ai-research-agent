from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import Document, DocumentSection


@pytest.mark.asyncio
async def test_document_sections_support_nested_hierarchy() -> None:
    async with async_session_factory() as session:
        document = Document(
            title=f"Section Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"section-test-{uuid4()}",
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        introduction = DocumentSection(
            document_id=document.id,
            title="Introduction",
            section_path="Introduction",
            section_level=1,
            section_index=0,
            section_metadata={},
        )

        session.add(introduction)
        await session.flush()

        background = DocumentSection(
            document_id=document.id,
            parent_section_id=introduction.id,
            title="Background",
            section_path="Introduction > Background",
            section_level=2,
            section_index=0,
            section_metadata={},
        )

        motivation = DocumentSection(
            document_id=document.id,
            parent_section_id=introduction.id,
            title="Motivation",
            section_path="Introduction > Motivation",
            section_level=2,
            section_index=1,
            section_metadata={},
        )

        session.add_all([background, motivation])
        await session.commit()

        result = await session.execute(
            select(DocumentSection)
            .where(
                DocumentSection.parent_section_id == introduction.id
            )
            .order_by(DocumentSection.section_index)
        )

        children = result.scalars().all()

        assert len(children) == 2
        assert children[0].title == "Background"
        assert children[1].title == "Motivation"

        assert children[0].parent_section_id == introduction.id
        assert children[1].parent_section_id == introduction.id

        await session.delete(document)
        await session.commit()