"""add document lifecycle

Revision ID: 32b24d0ab7d2
Revises: 86b2f3d39bfd
Create Date: 2026-09-03 15:36:54.605284

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '32b24d0ab7d2'
down_revision: str | Sequence[str] | None = '86b2f3d39bfd'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add document lifecycle fields."""

    op.add_column(
        "documents",
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "processing_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "processing_completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "failed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "processing_attempt",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "last_error",
            sa.Text(),
            nullable=True,
        ),
    )

    # Existing documents were already successfully ingested.
    op.execute(
        """
        UPDATE documents
        SET status = 'READY',
            processing_attempt = 1
        WHERE status IS NULL
        """
    )

    op.alter_column(
        "documents",
        "status",
        nullable=False,
    )

    op.alter_column(
        "documents",
        "processing_attempt",
        nullable=False,
    )

    op.create_index(
        "ix_documents_status",
        "documents",
        ["status"],
    )


def downgrade() -> None:
    """Remove document lifecycle fields."""

    op.drop_index(
        "ix_documents_status",
        table_name="documents",
    )

    op.drop_column("documents", "last_error")
    op.drop_column("documents", "processing_attempt")
    op.drop_column("documents", "failed_at")
    op.drop_column("documents", "processing_completed_at")
    op.drop_column("documents", "processing_started_at")
    op.drop_column("documents", "status")