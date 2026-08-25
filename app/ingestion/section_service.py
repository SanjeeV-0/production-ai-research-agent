from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import DocumentSection
from app.core.repositories.document import DocumentRepository
from app.ingestion.section_builder import SectionNode
from app.ingestion.section_map import SectionMap


class SectionService:
    """Persists document sections and builds their section map."""

    def __init__(self, session: AsyncSession) -> None:
        self.repository = DocumentRepository(session)

    async def persist_sections(
        self,
        document_id: UUID,
        nodes: list[SectionNode],
    ) -> SectionMap:
        """Persist section nodes and return their path-to-ID map."""
        section_map = SectionMap()

        for node in nodes:
            parent_section_id = None

            if node.parent_path is not None:
                parent_section_id = section_map.get(
                    node.parent_path
                )

            section = DocumentSection(
                document_id=document_id,
                parent_section_id=parent_section_id,
                title=node.title,
                section_path=node.path,
                section_level=node.level,
                section_index=node.index,
                section_metadata={},
            )

            await self.repository.create_section(section)

            section_map.add(
                node.path,
                section.id,
            )

        return section_map