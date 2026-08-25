from uuid import UUID


class SectionMap:
    """Maps logical section paths to persisted section IDs."""

    def __init__(self) -> None:
        self._mapping: dict[str, UUID] = {}

    def add(
        self,
        section_path: str,
        section_id: UUID,
    ) -> None:
        """Associate a section path with its database ID."""
        if section_path in self._mapping:
            raise ValueError(
                f"Section path already exists: {section_path}"
            )

        self._mapping[section_path] = section_id

    def get(self, section_path: str) -> UUID:
        """Return the database ID for a section path."""
        try:
            return self._mapping[section_path]
        except KeyError as exc:
            raise KeyError(
                f"Unknown section path: {section_path}"
            ) from exc

    def __contains__(self, section_path: str) -> bool:
        """Return whether a section path exists."""
        return section_path in self._mapping

    def __len__(self) -> int:
        """Return the number of mapped sections."""
        return len(self._mapping)