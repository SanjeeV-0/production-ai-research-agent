"""add chunk embeddings

Revision ID: 7403624356ba
Revises: 73ec85f9f102
Create Date: 2026-08-25 21:48:40.864239

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7403624356ba"
down_revision: str | Sequence[str] | None = "73ec85f9f102"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding",
            Vector(384),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column(
        "document_chunks",
        "embedding",
    )