from datetime import date

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """Validated representation of a document entering the ingestion pipeline."""

    title: str = Field(min_length=1, max_length=500)
    authors: str | None = None
    source: str | None = None
    publication_date: date | None = None
    document_type: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)