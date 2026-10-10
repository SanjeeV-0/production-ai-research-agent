"""add file blobs table

Revision ID: ff969effa790
Revises: e107da913094
Create Date: 2026-10-10 16:55:16.044555

Introduces `file_blobs`: one canonical row per RAW-BYTE content hash, shared
by any number of `StoredFile` rows (across any number of document versions
or logical documents). This is a FORWARD-COMPATIBLE change for existing
data: every pre-existing `StoredFile` row keeps its current, already-working
UUID-addressed `storage_key` completely unchanged -- this migration never
moves, renames, or touches a single physical file. It only backfills one
`file_blobs` row per DISTINCT existing `stored_files.content_hash`, pointing
at whichever storage key that content already lives at, so the new
`stored_files.content_hash -> file_blobs.content_hash` foreign key can be
added without requiring any data loss, re-ingestion, or physical file
migration. Only NEW uploads going forward get content-addressed storage
keys (see `app.storage.keys.build_blob_storage_key`).

Precondition checked before adding the foreign key: every existing
`stored_files.content_hash` value must have a corresponding `file_blobs`
row once the backfill runs. The backfill step guarantees this by
construction (it derives `file_blobs` directly from the distinct hashes
already present in `stored_files`), so no row can be orphaned -- this is
verified by an explicit `SELECT` guard before `upgrade()` proceeds to add
the FK, and the migration aborts with a clear error instead of silently
leaving an inconsistent schema if that invariant somehow does not hold.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ff969effa790"
down_revision: str | Sequence[str] | None = "e107da913094"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "file_blobs",
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=1000), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("content_hash"),
        sa.UniqueConstraint("storage_key", name="uq_file_blobs_storage_key"),
    )

    op.create_index(
        op.f("ix_file_blobs_status"),
        "file_blobs",
        ["status"],
        unique=False,
    )

    connection = op.get_bind()

    # Backfill: one file_blobs row per DISTINCT existing content_hash,
    # using the storage_key/size_bytes/created_at of the first stored_files
    # row observed for that hash. `DISTINCT ON` picks a single deterministic
    # row per hash without needing a GROUP BY aggregate for storage_key.
    connection.execute(
        sa.text(
            """
            INSERT INTO file_blobs
                (content_hash, storage_key, size_bytes, status, created_at, ready_at)
            SELECT DISTINCT ON (content_hash)
                content_hash, storage_key, size_bytes, 'READY', created_at, created_at
            FROM stored_files
            ORDER BY content_hash, created_at ASC
            """
        )
    )

    # Precondition check: every stored_files.content_hash must now resolve
    # to a file_blobs row, or the FK below would fail to apply anyway --
    # checked explicitly here so the failure is a clear, intentional
    # application error rather than an opaque constraint-violation
    # traceback, and so this migration never silently proceeds in an
    # inconsistent state.
    orphaned = connection.execute(
        sa.text(
            """
            SELECT count(*) FROM stored_files sf
            LEFT JOIN file_blobs fb ON fb.content_hash = sf.content_hash
            WHERE fb.content_hash IS NULL
            """
        )
    ).scalar_one()

    if orphaned:
        raise RuntimeError(
            f"{orphaned} stored_files row(s) have no corresponding file_blobs row after "
            "backfill; refusing to add the foreign key. This should be unreachable given "
            "the backfill above -- investigate before re-running this migration."
        )

    op.create_foreign_key(
        "fk_stored_files_content_hash_file_blobs",
        "stored_files",
        "file_blobs",
        ["content_hash"],
        ["content_hash"],
    )

    # storage_key is no longer globally unique on stored_files -- it is now
    # a denormalized copy of file_blobs.storage_key, and multiple
    # stored_files rows legitimately share one storage_key once they
    # reference the same blob.
    op.drop_constraint(
        "stored_files_storage_key_key",
        "stored_files",
        type_="unique",
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.create_unique_constraint(
        "stored_files_storage_key_key",
        "stored_files",
        ["storage_key"],
    )

    op.drop_constraint(
        "fk_stored_files_content_hash_file_blobs",
        "stored_files",
        type_="foreignkey",
    )

    op.drop_index(
        op.f("ix_file_blobs_status"),
        table_name="file_blobs",
    )

    op.drop_table("file_blobs")
