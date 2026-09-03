"""add stored file model

Revision ID: 7991853c5dd5
Revises: 9f0853bbd01b
Create Date: 2026-09-03 20:00:11.578519

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7991853c5dd5"
down_revision: str | Sequence[str] | None = "9f0853bbd01b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "stored_files",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column(
            "original_filename",
            sa.String(length=500),
            nullable=False,
        ),
        sa.Column(
            "content_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "storage_key",
            sa.String(length=1000),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )

    op.create_index(
        op.f("ix_stored_files_content_hash"),
        "stored_files",
        ["content_hash"],
        unique=False,
    )

    op.create_index(
        op.f("ix_stored_files_document_id"),
        "stored_files",
        ["document_id"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_stored_files_document_id"),
        table_name="stored_files",
    )

    op.drop_index(
        op.f("ix_stored_files_content_hash"),
        table_name="stored_files",
    )

    op.drop_table("stored_files")