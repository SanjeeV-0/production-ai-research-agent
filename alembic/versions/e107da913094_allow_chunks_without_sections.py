"""allow chunks without sections

Revision ID: e107da913094
Revises: 7991853c5dd5
Create Date: 2026-09-29 13:55:33.160072

"""

from collections.abc import Sequence

from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e107da913094"
down_revision: str | Sequence[str] | None = "7991853c5dd5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "document_chunks",
        "section_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "document_chunks",
        "section_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
