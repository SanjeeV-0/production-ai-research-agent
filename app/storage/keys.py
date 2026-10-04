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
    """
    return f"documents/{logical_document_id}/{document_version_id}/{file_id}"
