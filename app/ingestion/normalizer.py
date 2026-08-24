import hashlib


def normalize_content(content: str) -> str:
    """Normalize document text before hashing and downstream processing."""
    return " ".join(content.split())


def calculate_content_hash(content: str) -> str:
    """Calculate a deterministic SHA-256 hash of normalized document content."""
    normalized_content = normalize_content(content)

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()