from uuid import UUID


def build_storage_key(
    logical_document_id: UUID,
    document_version_id: UUID,
    file_id: UUID,
) -> str:
    """Build an immutable storage key for a document version's file."""
    return f"documents/{logical_document_id}/{document_version_id}/{file_id}"
