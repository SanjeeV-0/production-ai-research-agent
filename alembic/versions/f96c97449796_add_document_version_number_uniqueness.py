"""add document version number uniqueness

Revision ID: f96c97449796
Revises: df205cc766b7
Create Date: 2026-10-10 16:55:40.000000

Adds `uq_documents_logical_version`: a unique constraint on
`(logical_document_id, version_number)`. This is the database-level backing
for safe concurrent version allocation (see
`app.core.services.document.DocumentService.ingest_as_new_version`'s
bounded retry-on-conflict loop) -- previously, `version_number` was assigned
purely in application code via an unprotected `SELECT max()+1`, with
nothing in the schema preventing two concurrent requests from allocating
the same number to two different versions of one logical document.

Precondition checked before adding the constraint: no existing
`(logical_document_id, version_number)` pair may already be duplicated. If
one is found, this migration aborts with a clear error identifying the
affected logical document(s) rather than silently renumbering or deleting
any version.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f96c97449796"
down_revision: str | Sequence[str] | None = "df205cc766b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    connection = op.get_bind()

    duplicates = connection.execute(
        sa.text(
            """
            SELECT logical_document_id, version_number
            FROM documents
            GROUP BY logical_document_id, version_number
            HAVING count(*) > 1
            """
        )
    ).all()

    if duplicates:
        raise RuntimeError(
            "Cannot add uq_documents_logical_version: "
            f"{len(duplicates)} (logical_document_id, version_number) pair(s) are "
            f"already duplicated: {list(duplicates)}. Reconcile these versions "
            "(e.g. renumber manually, preserving created_at order) before "
            "re-running this migration."
        )

    op.create_unique_constraint(
        "uq_documents_logical_version",
        "documents",
        ["logical_document_id", "version_number"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_constraint(
        "uq_documents_logical_version",
        "documents",
        type_="unique",
    )
