import hashlib
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document, DocumentPage, DocumentStatus, StoredFile
from app.core.repositories.document import DocumentRepository
from app.core.repositories.stored_file import StoredFileRepository
from app.core.services.document import DocumentService
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.chunk_service import ChunkService
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.resolver import resolve_loader_for_filename
from app.ingestion.schemas import DocumentInput
from app.ingestion.section_builder import SectionBuilder
from app.ingestion.section_service import SectionService
from app.ingestion.semantic_shredder import shred_semantically
from app.ingestion.size_guard import apply_size_guard
from app.ingestion.structure_extractor import StructureExtractor
from app.storage.interface import FileStorage
from app.storage.keys import build_storage_key


class IngestionService:
    """Orchestrates the document ingestion workflow."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        file_storage: FileStorage,
    ) -> None:
        self.session = session
        self.file_storage = file_storage
        self.repository = DocumentRepository(session)
        self.stored_file_repository = StoredFileRepository(session)
        self.document_service = DocumentService(session)
        self.structure_extractor = StructureExtractor()
        self.section_builder = SectionBuilder()
        self.section_service = SectionService(session)
        self.chunk_service = ChunkService(
            session,
            embedding_provider=embedding_provider,
        )
        self.embedding_provider = embedding_provider

    async def ingest_file(
        self,
        path: Path,
        loader: DocumentLoader,
        title: str,
        document_type: str,
        logical_document_id: UUID | None = None,
        source: str | None = None,
        original_filename: str | None = None,
    ) -> Document:
        file_bytes = path.read_bytes()
        pages = loader.load(path)

        combined_content = "\n\n".join(page.content for page in pages)

        document, created = await self.document_service.ingest_document(
            DocumentInput(
                title=title,
                source=source,
                document_type=document_type,
                content=combined_content,
            ),
            logical_document_id=logical_document_id,
        )

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if not created:
            return document

        file_id = uuid4()

        storage_key = build_storage_key(
            logical_document_id=document.logical_document_id,
            document_version_id=document.id,
            file_id=file_id,
        )

        await self.file_storage.store(
            file_bytes,
            storage_key,
        )

        try:
            stored_file = StoredFile(
                id=file_id,
                document_id=document.id,
                original_filename=original_filename or path.name,
                content_hash=file_hash,
                size_bytes=len(file_bytes),
                storage_key=storage_key,
            )

            self.session.add(stored_file)

            await self.session.commit()

        except Exception:
            await self.session.rollback()

            try:
                await self.file_storage.delete(storage_key)
            except Exception:
                # Preserve the original DB/application failure.
                pass

            raise

        document = await self.document_service.mark_processing(document)

        await self.session.commit()

        return await self._process_document(
            document=document,
            pages=pages,
        )

    async def retry_document(
        self,
        document_id: UUID,
        loader: DocumentLoader | None = None,
    ) -> Document:
        """Retry a FAILED document version, reusing its existing StoredFile
        and physical file -- no new version or stored file is created.

        `loader` is optional: when not supplied, it is resolved from the
        existing StoredFile's `original_filename` (the same loader the
        original upload would have used), since that filename is already
        known to the service and does not need to come from the caller.
        """

        document = await self.repository.get_by_id(document_id)

        if document is None:
            raise ValueError(f"Document {document_id} not found.")

        if document.status != DocumentStatus.FAILED:
            raise ValueError("Only FAILED documents can be retried.")

        stored_file = await self.stored_file_repository.get_by_document_id(document.id)

        if stored_file is None:
            raise ValueError(f"No stored file found for document {document.id}.")

        if loader is None:
            loader = resolve_loader_for_filename(stored_file.original_filename)

        file_bytes = await self.file_storage.retrieve(stored_file.storage_key)

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(stored_file.original_filename).suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(file_bytes)
                temporary_path = Path(temporary_file.name)

            pages = loader.load(temporary_path)

            document = await self.document_service.mark_processing(document)

            await self.session.commit()

            return await self._process_document(
                document=document,
                pages=pages,
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def _process_document(
        self,
        document: Document,
        pages: list,
    ) -> Document:
        document_id = document.id

        try:
            page_ids: dict[int, UUID] = {}

            for page in pages:
                document_page = DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    content=page.content,
                )

                await self.repository.create_page(document_page)

                page_ids[page.page_number] = document_page.id

            structural_units = self.structure_extractor.extract(pages)

            section_nodes = self.section_builder.build(structural_units)

            section_map = await self.section_service.persist_sections(
                document.id,
                section_nodes,
            )

            semantic_units = shred_semantically(
                structural_units,
                embedding_provider=self.embedding_provider,
                threshold=0.7,
            )

            child_chunks = apply_size_guard(
                semantic_units,
                section_map=section_map,
                max_tokens=500,
            )

            await self.chunk_service.persist_chunks(
                document_id=document.id,
                page_ids=page_ids,
                chunks=child_chunks,
            )

            document = await self.document_service.mark_ready(document)

            await self.session.commit()

            return document

        except Exception as exc:
            await self.session.rollback()

            document = await self.repository.get_by_id(document_id)

            if document is None:
                raise RuntimeError("Document disappeared after ingestion failure.") from exc

            await self.document_service.mark_failed(
                document,
                str(exc),
            )

            await self.session.commit()

            # Attach the persisted FAILED version to the exception itself
            # (without changing its type or message) so a caller such as
            # the HTTP layer can report that version back to the client
            # instead of losing it behind a generic error.
            exc.failed_document = document

            raise
