"""SQLAlchemy ORM models -- the canonical schema definition.

Note: some invariants are enforced at the DATABASE level by Alembic
migrations rather than by anything visible on these model classes --
notably the partial unique index that guarantees at most one `is_current`
row per `logical_document_id`
(`alembic/versions/9f0853bbd01b_add_document_versioning.py`,
`ix_documents_current_version ... WHERE is_current = true`) and the
`uq_documents_logical_content_hash` unique constraint that backs ingestion
idempotency. Always cross-check `alembic/versions/` when reasoning about
what this schema actually guarantees.
"""

from datetime import UTC, date, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class DocumentStatus(StrEnum):
    """Lifecycle states for an ingested document."""

    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class Document(Base):
    """One ROW PER INGESTED VERSION -- despite the class name, this is not
    "one row per logical document". `logical_document_id` is the stable
    identity shared across every version of the same source document;
    `id` (this row's own primary key) is what the rest of the codebase
    calls a "version id". See `app.core.services.document.DocumentService`
    for the version lifecycle this table participates in.
    """

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    authors: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    publication_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    document_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    logical_document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    # SHA-256 of whitespace-normalized EXTRACTED TEXT content (see
    # app.ingestion.normalizer), not of the raw uploaded file bytes --
    # StoredFile.content_hash covers the raw bytes separately. Combined with
    # logical_document_id, this is what the application uses to decide
    # whether an upload is a true no-op duplicate (see DocumentService.
    # ingest_document); the DB-level uniqueness is the
    # uq_documents_logical_content_hash constraint added in migration
    # 9f0853bbd01b, not visible directly on this column definition.
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    version_number: Mapped[int] = mapped_column(
        nullable=False,
    )

    # At most one row per logical_document_id may be True -- enforced by the
    # partial unique index ix_documents_current_version (migration
    # 9f0853bbd01b), not by anything expressible on this column alone.
    is_current: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        index=True,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        String(20),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        index=True,
    )

    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    processing_attempt: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
    )

    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    document_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    sections: Mapped[list["DocumentSection"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class StoredFile(Base):
    """Metadata for the ONE physical file backing a `Document` version
    (`document_id` is unique -- a version has at most one StoredFile, ever).
    The actual bytes live outside the database, via `app.storage.FileStorage`
    at `storage_key`; this row is how the application finds them again.
    """

    __tablename__ = "stored_files"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # Deliberately NO `ondelete="CASCADE"` here (unlike DocumentPage/
    # DocumentSection/DocumentChunk below) -- DocumentDeletionService must
    # delete this row explicitly BEFORE deleting the Document row, or the FK
    # would block the delete. See app.core.services.document_deletion.
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    size_bytes: Mapped[int] = mapped_column(
        nullable=False,
    )

    storage_key: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        unique=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class DocumentPage(Base):
    """Extracted page-level content belonging to a document."""

    __tablename__ = "document_pages"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    page_number: Mapped[int] = mapped_column(
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    document: Mapped["Document"] = relationship(
        back_populates="pages",
    )
    chunk_mappings: Mapped[list["ChunkPageMap"]] = relationship(
        back_populates="document_page",
        cascade="all, delete-orphan",
    )


class DocumentSection(Base):
    """Logical section within a document."""

    __tablename__ = "document_sections"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    parent_section_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    section_path: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
    )

    section_level: Mapped[int] = mapped_column(
        nullable=False,
    )

    section_index: Mapped[int] = mapped_column(
        nullable=False,
    )

    section_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    document: Mapped["Document"] = relationship(
        back_populates="sections",
    )

    parent_section: Mapped["DocumentSection | None"] = relationship(
        back_populates="child_sections",
        remote_side="DocumentSection.id",
    )

    child_sections: Mapped[list["DocumentSection"]] = relationship(
        back_populates="parent_section",
        cascade="all, delete-orphan",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="section",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """A searchable chunk derived from document content."""

    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    chunk_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    document: Mapped["Document"] = relationship(
        back_populates="chunks",
    )

    page_mappings: Mapped[list["ChunkPageMap"]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
    )
    # Nullable (migration e107da913094_allow_chunks_without_sections) so
    # headingless documents -- which have no DocumentSection at all -- can
    # still produce retrievable chunks. search_similar_chunks relies on this
    # being nullable: it outer-joins DocumentSection specifically so a
    # section_id IS NULL chunk is still returned, not silently excluded.
    section_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    section: Mapped["DocumentSection"] = relationship(
        back_populates="chunks",
    )
    # Dimension (384) is fixed to match the default EMBEDDING_MODEL
    # (sentence-transformers/all-MiniLM-L6-v2) and is NOT derived from
    # Settings at runtime -- switching to a model with a different output
    # dimension requires a migration to alter this column, in addition to
    # re-embedding every existing chunk (see app.embeddings.sentence_transformer).
    # Indexed via an HNSW index (migration 86b2f3d39bfd) using
    # vector_cosine_ops, matching the cosine_distance() query in
    # DocumentRepository.search_similar_chunks.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True,
    )


class ChunkPageMap(Base):
    """Maps a document chunk to one of its source pages."""

    __tablename__ = "chunk_page_map"

    chunk_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )

    document_page_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("document_pages.id", ondelete="CASCADE"),
        primary_key=True,
    )

    chunk: Mapped["DocumentChunk"] = relationship(
        back_populates="page_mappings",
    )

    document_page: Mapped["DocumentPage"] = relationship(
        back_populates="chunk_mappings",
    )
