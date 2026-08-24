from abc import ABC, abstractmethod
from pathlib import Path


class DocumentLoader(ABC):
    """Interface for converting source files into text."""

    @abstractmethod
    def load(self, path: Path) -> str:
        """Load and return normalized source text."""