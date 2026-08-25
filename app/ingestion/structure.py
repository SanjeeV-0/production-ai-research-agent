from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class UnitType(StrEnum):
    """Types of structural content extracted from a document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"


@dataclass(frozen=True)
class TableData:
    """Structured representation of a document table."""

    table_id: str
    title: str | None
    headers: list[str]
    rows: list[list[str]]
    page_numbers: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class StructuralUnit:
    """A logical piece of content extracted from a document."""

    unit_type: UnitType
    content: str
    page_numbers: list[int]
    section_path: str
    section_level: int
    section_index: int
    table: TableData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)