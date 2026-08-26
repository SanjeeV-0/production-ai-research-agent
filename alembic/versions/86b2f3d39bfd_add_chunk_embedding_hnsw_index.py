"""add chunk embedding hnsw index

Revision ID: 86b2f3d39bfd
Revises: 7403624356ba
Create Date: 2026-08-26 14:54:47.820196

"""


from typing import Sequence, Union

from alembic import op


revision: str = "86b2f3d39bfd"
down_revision: Union[str, Sequence[str], None] = "7403624356ba"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={
            "embedding": "vector_cosine_ops",
        },
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_document_chunks_embedding_hnsw",
        table_name="document_chunks",
    )