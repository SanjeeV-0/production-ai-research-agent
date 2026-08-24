from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedPage:
    """Extracted content and provenance for one document page."""

    page_number: int
    content: str


class DocumentLoader(ABC):
    """Interface for converting source files into structured pages."""

    @abstractmethod
    def load(self, path: Path) -> list[LoadedPage]:
        """Load a source document and preserve page-level content."""