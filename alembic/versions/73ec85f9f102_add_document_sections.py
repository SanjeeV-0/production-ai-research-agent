"""add document sections

Revision ID: 73ec85f9f102
Revises: 7d01a8f0608b
Create Date: 2026-08-25 13:55:34.073866

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '73ec85f9f102'
down_revision: str | Sequence[str] | None = '7d01a8f0608b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "document_sections",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("parent_section_id", sa.UUID(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("section_path", sa.String(length=2000), nullable=False),
        sa.Column("section_level", sa.Integer(), nullable=False),
        sa.Column("section_index", sa.Integer(), nullable=False),
        sa.Column(
            "section_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_section_id"],
            ["document_sections.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_document_sections_document_id",
        "document_sections",
        ["document_id"],
        unique=False,
    )

    op.create_index(
        "ix_document_sections_parent_section_id",
        "document_sections",
        ["parent_section_id"],
        unique=False,
    )

    op.add_column(
        "document_chunks",
        sa.Column("section_id", sa.UUID(), nullable=True),
    )

    op.create_index(
        "ix_document_chunks_section_id",
        "document_chunks",
        ["section_id"],
        unique=False,
    )

    op.execute(
        """
        INSERT INTO document_sections (
            id,
            document_id,
            parent_section_id,
            title,
            section_path,
            section_level,
            section_index,
            section_metadata
        )
        SELECT
            gen_random_uuid(),
            d.id,
            NULL,
            'Legacy Section',
            'Legacy Section',
            1,
            0,
            '{}'::jsonb
        FROM documents AS d
        WHERE EXISTS (
            SELECT 1
            FROM document_chunks AS c
            WHERE c.document_id = d.id
        )
        """
    )

    op.execute(
        """
        UPDATE document_chunks AS c
        SET section_id = s.id
        FROM document_sections AS s
        WHERE s.document_id = c.document_id
          AND s.title = 'Legacy Section'
        """
    )

    op.alter_column(
        "document_chunks",
        "section_id",
        nullable=False,
    )

    op.create_foreign_key(
        "fk_document_chunks_section_id",
        "document_chunks",
        "document_sections",
        ["section_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_document_chunks_section_id",
        "document_chunks",
        type_="foreignkey",
    )

    op.drop_index(
        "ix_document_chunks_section_id",
        table_name="document_chunks",
    )

    op.drop_column(
        "document_chunks",
        "section_id",
    )

    op.drop_index(
        "ix_document_sections_parent_section_id",
        table_name="document_sections",
    )

    op.drop_index(
        "ix_document_sections_document_id",
        table_name="document_sections",
    )

    op.drop_table("document_sections")