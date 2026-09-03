from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.models import Document, DocumentSection
from app.ingestion.section_builder import SectionBuilder
from app.ingestion.section_service import SectionService
from app.ingestion.structure import StructuralUnit, UnitType


def _unit(
    section_path: str,
    section_level: int,
    section_index: int,
) -> StructuralUnit:
    return StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content="content",
        page_numbers=[1],
        section_path=section_path,
        section_level=section_level,
        section_index=section_index,
    )


@pytest.mark.asyncio
async def test_persist_sections_builds_parent_relationships() -> None:
    async with async_session_factory() as session:
        document = Document(
            title=f"Section Test {uuid4()}",
            document_type="research_paper",
            content_hash=f"section-test-{uuid4()}",
            logical_document_id=uuid4(),
            version_number=1,
            is_current=True,
            document_metadata={},
        )

        session.add(document)
        await session.flush()

        units = [
            _unit("Results", 1, 0),
            _unit("Results > Retrieval", 2, 0),
            _unit("Results > Evaluation", 2, 1),
        ]

        nodes = SectionBuilder().build(units)

        service = SectionService(session)

        section_map = await service.persist_sections(
            document.id,
            nodes,
        )

        await session.commit()

        results = await session.execute(
            select(DocumentSection)
            .where(
                DocumentSection.document_id == document.id
            )
            .order_by(
                DocumentSection.section_level,
                DocumentSection.section_index,
            )
        )

        sections = results.scalars().all()

        assert len(sections) == 3

        root = sections[0]
        retrieval = sections[1]
        evaluation = sections[2]

        assert root.parent_section_id is None

        assert retrieval.parent_section_id == root.id
        assert evaluation.parent_section_id == root.id

        assert section_map.get("Results") == root.id
        assert (
            section_map.get("Results > Retrieval")
            == retrieval.id
        )
        assert (
            section_map.get("Results > Evaluation")
            == evaluation.id
        )

        await session.delete(document)
        await session.commit()