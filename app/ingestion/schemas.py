from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import Document, DocumentStatus


class DocumentInput(BaseModel):
    """Validated representation of a document entering the ingestion pipeline."""

    title: str = Field(min_length=1, max_length=500)
    authors: str | None = None
    source: str | None = None
    publication_date: date | None = None
    document_type: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)


class DocumentVersionResponse(BaseModel):
    """Explicit API representation of a single ingested document version.

    Deliberately excludes SQLAlchemy-internal and relationship fields
    (`document_metadata`, `pages`, `chunks`, `sections`, `content_hash`,
    `authors`, `publication_date`, ...) that exist on the `Document` ORM
    model but are not part of the document-management HTTP contract.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    logical_document_id: UUID
    version_number: int
    is_current: bool
    status: DocumentStatus
    title: str
    document_type: str
    source: str | None
    processing_attempt: int
    created_at: datetime
    updated_at: datetime
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    failed_at: datetime | None
    last_error: str | None


class IngestOutcome(StrEnum):
    """What `POST /documents` actually did with a given upload.

    CREATED: no matching normalized content existed anywhere -- a brand new
    logical document (server-generated ID) and its version 1 were created.

    DUPLICATE: the upload's normalized extracted-text content already
    matched an existing version (globally, regardless of which logical
    document it belongs to, its filename, or its category) -- no new
    logical document, version, section, chunk, or embedding was created.
    `version` in the response is the pre-existing matched version, and
    `updated_metadata_fields` lists which of its fields were updated from
    the submitted (non-empty) metadata, if any.

    NEW_VERSION: the upload was submitted against an explicitly chosen
    existing `logical_document_id` and its content did not already match
    any existing version under that same logical document -- a new version
    row was created under it.
    """

    CREATED = "created"
    DUPLICATE = "duplicate"
    NEW_VERSION = "new_version"


class DocumentUploadResponse(BaseModel):
    """Response shape for `POST /documents`, replacing the previous bare
    `DocumentVersionResponse` so callers can distinguish a true new upload
    from a detected duplicate or a new version of an existing document --
    see `IngestOutcome` for exactly what each value means."""

    outcome: IngestOutcome
    version: DocumentVersionResponse
    updated_metadata_fields: list[str] = Field(default_factory=list)


class LogicalDocumentResponse(BaseModel):
    """API representation of one logical document: a stable identity
    (`logical_document_id`) plus its current version, if one exists.

    Reuses the canonical `DocumentVersionResponse` for `current_version`
    (same `id` field as POST /documents) rather than a near-duplicate
    schema, so there is exactly one version-shaped response contract.

    A logical document with no current version (e.g. its only version(s)
    are still PROCESSING or FAILED) is still represented, with
    `current_version=None` -- version history remains independently
    available via GET /documents/{logical_document_id}/versions.
    """

    logical_document_id: UUID
    current_version: DocumentVersionResponse | None

    @classmethod
    def from_current_version(
        cls,
        logical_document_id: UUID,
        current_version: Document | None,
    ) -> "LogicalDocumentResponse":
        return cls(
            logical_document_id=logical_document_id,
            current_version=(
                DocumentVersionResponse.model_validate(current_version)
                if current_version is not None
                else None
            ),
        )
