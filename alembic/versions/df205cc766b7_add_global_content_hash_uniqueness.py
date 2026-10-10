"""add global content hash uniqueness

Revision ID: df205cc766b7
Revises: ff969effa790
Create Date: 2026-10-10 16:55:30.000000

Replaces the per-logical-document `uq_documents_logical_content_hash`
constraint (logical_document_id, content_hash) with a GLOBAL unique
constraint on `documents.content_hash` alone -- the schema-level backing for
"detect a normalized-content duplicate regardless of which logical document,
filename, or category it was first attached to" (see
`app.core.services.document.DocumentService.ingest_as_new_or_duplicate`).

Precondition checked before adding the new constraint: no two existing
`documents` rows may already share a `content_hash` across DIFFERENT
logical documents (within the same logical document, the old constraint
already guaranteed at most one row per hash, so only a cross-document
collision could violate the new, stricter constraint). If any such
collision exists, this migration aborts with a clear error identifying the
colliding hash(es) rather than silently deleting or merging data -- an
operator must decide how to reconcile those rows (e.g. which version is
authoritative) before this migration can proceed.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "df205cc766b7"
down_revision: str | Sequence[str] | None = "ff969effa790"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    connection = op.get_bind()

    colliding_hashes = connection.execute(
        sa.text(
            """
            SELECT content_hash
            FROM documents
            GROUP BY content_hash
            HAVING count(DISTINCT logical_document_id) > 1
            """
        )
    ).scalars().all()

    if colliding_hashes:
        raise RuntimeError(
            "Cannot add a global unique constraint on documents.content_hash: "
            f"{len(colliding_hashes)} content_hash value(s) are already shared across "
            f"more than one logical_document_id: {list(colliding_hashes)}. Reconcile "
            "these rows (e.g. decide which logical document should own that content, "
            "or merge manually) before re-running this migration."
        )

    op.drop_constraint(
        "uq_documents_logical_content_hash",
        "documents",
        type_="unique",
    )

    op.create_unique_constraint(
        "uq_documents_content_hash",
        "documents",
        ["content_hash"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_constraint(
        "uq_documents_content_hash",
        "documents",
        type_="unique",
    )

    op.create_unique_constraint(
        "uq_documents_logical_content_hash",
        "documents",
        ["logical_document_id", "content_hash"],
    )
