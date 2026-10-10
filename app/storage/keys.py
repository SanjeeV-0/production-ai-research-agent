from uuid import UUID


def build_storage_key(
    logical_document_id: UUID,
    document_version_id: UUID,
    file_id: UUID,
) -> str:
    """Build an immutable storage key for a document version's file.

    Deliberately built ONLY from server-generated UUIDs, never from the
    user-supplied original filename -- the filename is stored purely as
    `StoredFile.original_filename` metadata. This means attacker/user
    controlled filename content (special characters, path separators,
    duplicate names across uploads) never reaches a filesystem path.

    Superseded by `build_blob_storage_key` for all NEW blobs (see
    `app.ingestion.blob_service`) -- kept only because existing `FileBlob`
    rows backfilled from pre-blob-model `StoredFile` rows
    (`alembic/versions/<add_file_blobs_table>.py`) still point at keys built
    by this function, and must keep resolving correctly.
    """
    return f"documents/{logical_document_id}/{document_version_id}/{file_id}"


def build_blob_storage_key(content_hash: str) -> str:
    """Build a deterministic, content-addressed storage key for a blob.

    Every upload with the same raw-byte `content_hash` maps to exactly the
    same key, which is what makes physical blob sharing possible -- two
    `StoredFile` rows (even across different logical documents) that
    reference the same `FileBlob.content_hash` always resolve to this same
    path. The two-character prefix directory keeps any one directory from
    accumulating one entry per distinct file ever uploaded.
    """
    return f"blobs/{content_hash[:2]}/{content_hash}"
