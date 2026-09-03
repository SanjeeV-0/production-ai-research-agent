"""add document versioning

Revision ID: 9f0853bbd01b
Revises: 32b24d0ab7d2
Create Date: 2026-09-03 17:18:52.119254

"""
from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9f0853bbd01b'
down_revision: str | Sequence[str] | None = '32b24d0ab7d2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "logical_document_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "version_number",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "is_current",
            sa.Boolean(),
            nullable=True,
        ),
    )

    # Existing documents become version 1 of their own logical document.
    connection = op.get_bind()

    documents_table = sa.table(
        "documents",
        sa.column(
            "id",
            postgresql.UUID(as_uuid=True),
        ),
        sa.column(
            "logical_document_id",
            postgresql.UUID(as_uuid=True),
        ),
        sa.column(
            "version_number",
            sa.Integer(),
        ),
        sa.column(
            "is_current",
            sa.Boolean(),
        ),
    )

    rows = connection.execute(
        sa.select(documents_table.c.id)
    ).fetchall()

    for row in rows:
        connection.execute(
            documents_table.update()
            .where(documents_table.c.id == row.id)
            .values(
                logical_document_id=uuid4(),
                version_number=1,
                is_current=True,
            )
        )

    # Remove the old globally-unique content hash index.
    op.drop_index(
        "ix_documents_content_hash",
        table_name="documents",
    )

    # Enforce idempotency within a logical document.
    op.create_unique_constraint(
        "uq_documents_logical_content_hash",
        "documents",
        ["logical_document_id", "content_hash"],
    )

    # Only one version may be current for a logical document.
    op.create_index(
        "ix_documents_current_version",
        "documents",
        ["logical_document_id"],
        unique=True,
        postgresql_where=sa.text("is_current = true"),
    )

    # Match the model's explicit indexes.
    op.create_index(
        "ix_documents_logical_document_id",
        "documents",
        ["logical_document_id"],
    )

    op.create_index(
        "ix_documents_is_current",
        "documents",
        ["is_current"],
    )

    op.alter_column(
        "documents",
        "logical_document_id",
        nullable=False,
    )

    op.alter_column(
        "documents",
        "version_number",
        nullable=False,
    )

    op.alter_column(
        "documents",
        "is_current",
        nullable=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_is_current",
        table_name="documents",
    )

    op.drop_index(
        "ix_documents_logical_document_id",
        table_name="documents",
    )

    op.drop_index(
        "ix_documents_current_version",
        table_name="documents",
    )

    op.drop_constraint(
        "uq_documents_logical_content_hash",
        "documents",
        type_="unique",
    )

    # Restore the old global content-hash uniqueness.
    op.create_index(
        "ix_documents_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
    )

    op.drop_column("documents", "is_current")
    op.drop_column("documents", "version_number")
    op.drop_column("documents", "logical_document_id")
 