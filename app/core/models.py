from datetime import date, datetime
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
    """Canonical metadata representation of an ingested research document."""

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
    content_hash: Mapped[str] = mapped_column(
    String(64),
    nullable=False,
    index=True,
)

    version_number: Mapped[int] = mapped_column(
    nullable=False,
)

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
    section_id: Mapped[UUID] = mapped_column(
    PostgreSQLUUID(as_uuid=True),
    ForeignKey("document_sections.id", ondelete="CASCADE"),
    nullable=False,
    index=True,
)
    section: Mapped["DocumentSection"] = relationship(
    back_populates="chunks",
)
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
   