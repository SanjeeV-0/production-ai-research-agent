import hashlib


def normalize_content(content: str) -> str:
    """Collapse all whitespace runs to single spaces before hashing.

    This means two uploads whose EXTRACTED TEXT differs only in whitespace
    (e.g. re-exporting the same PDF with different line wrapping) hash
    identically and are treated as the same content by
    DocumentService.ingest_document's idempotency check.
    """
    return " ".join(content.split())


def calculate_content_hash(content: str) -> str:
    """Deterministic SHA-256 hash of normalized EXTRACTED document content.

    This is `Document.content_hash` -- distinct from `StoredFile.content_hash`,
    which hashes the raw uploaded file bytes. Two different files that
    extract to the same text (e.g. a re-saved PDF) produce the same
    Document.content_hash but different StoredFile.content_hash values.
    """
    normalized_content = normalize_content(content)

    return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
