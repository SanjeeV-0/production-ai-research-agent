from datetime import date, datetime
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
