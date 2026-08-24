import pytest
from pydantic import ValidationError

from app.ingestion.schemas import DocumentInput


def test_document_input_accepts_valid_document() -> None:
    document = DocumentInput(
        title="Retrieval-Augmented Generation",
        authors="Test Author",
        source="research-paper",
        document_type="research_paper",
        content="This is test document content.",
    )

    assert document.title == "Retrieval-Augmented Generation"
    assert document.document_type == "research_paper"


def test_document_input_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        DocumentInput(
            title="Test Document",
            document_type="research_paper",
            content="",
        )