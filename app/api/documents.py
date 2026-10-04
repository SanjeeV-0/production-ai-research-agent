"""Document-management HTTP endpoints.

This module is a thin FastAPI boundary over the existing ingestion domain
services -- it contains no ingestion/lifecycle logic of its own. All actual
work (content-hash deduplication, versioning, chunking, embedding, status
transitions) is performed by `app.ingestion.service.IngestionService`,
reused unchanged.

Only `POST /documents`, `GET /documents`,
`GET /documents/{logical_document_id}`,
`GET /documents/{logical_document_id}/versions`, and
`POST /documents/{logical_document_id}/versions/{version_id}/retry` are
implemented so far. The remaining approved routes (version deletion,
logical-document deletion) are intentionally not implemented yet.
"""

import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.dependencies import get_document_repository, get_ingestion_service
from app.core.models import Document
from app.core.repositories.document import DocumentRepository
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.resolver import resolve_loader_for_filename
from app.ingestion.schemas import DocumentVersionResponse, LogicalDocumentResponse
from app.ingestion.service import IngestionService

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


def _resolve_loader(filename: str) -> DocumentLoader:
    """Pick the registered loader implementation for a file, by extension."""

    try:
        return resolve_loader_for_filename(filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post(
    "",
    response_model=DocumentVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    ingestion_service: Annotated[
        IngestionService,
        Depends(get_ingestion_service),
    ],
    file: Annotated[UploadFile, File()],
    document_type: Annotated[str, Form(min_length=1, max_length=100)],
    title: Annotated[str | None, Form(max_length=500)] = None,
    source: Annotated[str | None, Form(max_length=1000)] = None,
    logical_document_id: Annotated[UUID | None, Form()] = None,
) -> DocumentVersionResponse:
    """Upload a file and ingest it via the existing IngestionService.

    This is a synchronous, in-request operation -- it mirrors the existing
    `IngestionService.ingest_file` flow exactly and does not introduce any
    background job, queue, or async task semantics.
    """

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    filename = file.filename or ""

    loader = _resolve_loader(filename)

    resolved_title = (title or "").strip() or Path(filename).stem or "Untitled Document"

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        document: Document = await ingestion_service.ingest_file(
            path=temporary_path,
            loader=loader,
            title=resolved_title,
            document_type=document_type,
            logical_document_id=logical_document_id,
            source=source,
            original_filename=filename,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # IngestionService persists a FAILED Document record (with
        # last_error/failed_at) before raising, and attaches that record to
        # the exception as `.failed_document`. A FAILED version is a normal,
        # trackable, retry-able lifecycle outcome -- not a request error --
        # so it is reported like any other successfully created version
        # rather than as a 422/500.
        document = getattr(exc, "failed_document", None)

        if document is None:
            # Nothing was ever persisted (e.g. the file could not even be
            # read/parsed) -- there is no version to report, so this is not
            # fabricated as a success and is not mislabeled as a 422
            # validation error either.
            raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return DocumentVersionResponse.model_validate(document)


@router.get(
    "",
    response_model=list[LogicalDocumentResponse],
)
async def list_documents(
    document_repository: Annotated[
        DocumentRepository,
        Depends(get_document_repository),
    ],
) -> list[LogicalDocumentResponse]:
    """List every logical document, each with its current version (if any).

    Each `logical_document_id` appears at most once, regardless of how many
    versions it has. Full version history remains available separately via
    GET /documents/{logical_document_id}/versions (not yet implemented).
    """

    logical_documents = await document_repository.list_logical_documents()

    return [
        LogicalDocumentResponse.from_current_version(logical_document_id, current_version)
        for logical_document_id, current_version in logical_documents
    ]


@router.get(
    "/{logical_document_id}",
    response_model=LogicalDocumentResponse,
)
async def get_document(
    logical_document_id: UUID,
    document_repository: Annotated[
        DocumentRepository,
        Depends(get_document_repository),
    ],
) -> LogicalDocumentResponse:
    """Return one logical document: its identity plus its current version.

    Same response shape as one item from GET /documents. Does not return
    version history -- that is GET /documents/{logical_document_id}/versions.
    """

    versions = await document_repository.get_by_logical_document_id(logical_document_id)

    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {logical_document_id}",
        )

    current_version = await document_repository.get_current_version(logical_document_id)

    return LogicalDocumentResponse.from_current_version(logical_document_id, current_version)


@router.get(
    "/{logical_document_id}/versions",
    response_model=list[DocumentVersionResponse],
)
async def list_document_versions(
    logical_document_id: UUID,
    document_repository: Annotated[
        DocumentRepository,
        Depends(get_document_repository),
    ],
) -> list[DocumentVersionResponse]:
    """Return every version of one logical document, newest first.

    Includes every lifecycle status (READY, FAILED, PROCESSING, UPLOADED) --
    not filtered to current or READY versions.
    """

    versions = await document_repository.get_by_logical_document_id(logical_document_id)

    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {logical_document_id}",
        )

    return [DocumentVersionResponse.model_validate(version) for version in versions]


@router.post(
    "/{logical_document_id}/versions/{version_id}/retry",
    response_model=DocumentVersionResponse,
)
async def retry_document_version(
    logical_document_id: UUID,
    version_id: UUID,
    ingestion_service: Annotated[
        IngestionService,
        Depends(get_ingestion_service),
    ],
    document_repository: Annotated[
        DocumentRepository,
        Depends(get_document_repository),
    ],
) -> DocumentVersionResponse:
    """Retry a FAILED document version via the existing IngestionService
    retry flow, reusing the same Document row, the same StoredFile, and the
    same physical file -- no new version or stored file is created.

    No request body is required: `version_id` already identifies the exact
    attempt to retry, and the loader is resolved internally by
    IngestionService from the existing StoredFile's filename.
    """

    version = await document_repository.get_by_id(version_id)

    if version is None or version.logical_document_id != logical_document_id:
        # Covers both "doesn't exist" and "belongs to a different logical
        # document" with the same 404, rather than revealing it exists
        # elsewhere.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document version not found: {version_id}",
        )

    try:
        document = await ingestion_service.retry_document(version_id)
    except Exception as exc:
        failed_document = getattr(exc, "failed_document", None)

        if failed_document is not None:
            # A normal, trackable, retry-able processing failure -- the
            # retried version is still reported, just as FAILED, exactly
            # like a first-time ingestion failure in POST /documents.
            document = failed_document
        elif isinstance(exc, ValueError):
            # IngestionService's own state guard: the targeted version is
            # not FAILED (existence was already confirmed above).
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        else:
            raise

    return DocumentVersionResponse.model_validate(document)
