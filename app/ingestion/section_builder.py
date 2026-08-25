from dataclasses import dataclass

from app.ingestion.structure import StructuralUnit


@dataclass(frozen=True)
class SectionNode:
    """Pure representation of a logical document section."""

    path: str
    title: str
    level: int
    index: int
    parent_path: str | None


class SectionBuilder:
    """Build a logical section hierarchy from structural units."""

    def build(
        self,
        units: list[StructuralUnit],
    ) -> list[SectionNode]:
        """Build unique sections in document appearance order."""
        sections: list[SectionNode] = []
        section_by_path: dict[str, SectionNode] = {}
        sibling_counters: dict[str | None, int] = {}

        for unit in units:
            path = unit.section_path

            if not path or path in section_by_path:
                continue

            parts = [
                part.strip()
                for part in path.split(">")
                if part.strip()
            ]

            if not parts:
                continue

            current_path_parts: list[str] = []

            for level, title in enumerate(parts, start=1):
                current_path_parts.append(title)
                current_path = " > ".join(
                    current_path_parts
                )

                if current_path in section_by_path:
                    continue

                parent_path = (
                    " > ".join(current_path_parts[:-1])
                    if len(current_path_parts) > 1
                    else None
                )

                index = sibling_counters.get(
                    parent_path,
                    0,
                )

                sibling_counters[parent_path] = index + 1

                node = SectionNode(
                    path=current_path,
                    title=title,
                    level=level,
                    index=index,
                    parent_path=parent_path,
                )

                sections.append(node)
                section_by_path[current_path] = node

        return sections