import re

from app.ingestion.loaders.base import LoadedPage
from app.ingestion.structure import StructuralUnit, UnitType

_HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)

_LIST_PATTERN = re.compile(
    r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)$"
)


class StructureExtractor:
    """Extract logical structural units from loaded document pages."""

    def extract(
        self,
        pages: list[LoadedPage],
    ) -> list[StructuralUnit]:
        """Convert loaded pages into structural units."""
        units: list[StructuralUnit] = []

        section_stack: list[tuple[int, str]] = []
        section_counters: dict[str, int] = {}

        for page in pages:
            paragraph_lines: list[str] = []

            for line in page.content.splitlines():
                stripped = line.strip()

                if not stripped:
                    units.extend(
                        self._flush_paragraph(
                            paragraph_lines=paragraph_lines,
                            page_number=page.page_number,
                            section_stack=section_stack,
                            section_counters=section_counters,
                        )
                    )
                    paragraph_lines.clear()
                    continue

                heading_match = _HEADING_PATTERN.match(stripped)

                if heading_match:
                    units.extend(
                        self._flush_paragraph(
                            paragraph_lines=paragraph_lines,
                            page_number=page.page_number,
                            section_stack=section_stack,
                            section_counters=section_counters,
                        )
                    )
                    paragraph_lines.clear()

                    level = len(heading_match.group(1))
                    title = heading_match.group(2)

                    while (
                        section_stack
                        and section_stack[-1][0] >= level
                    ):
                        section_stack.pop()

                    section_stack.append(
                        (level, title)
                    )

                    units.append(
                        StructuralUnit(
                            unit_type=UnitType.HEADING,
                            content=title,
                            page_numbers=[page.page_number],
                            section_path=self._section_path(
                                section_stack
                            ),
                            section_level=level,
                            section_index=0,
                        )
                    )

                    continue

                list_match = _LIST_PATTERN.match(stripped)

                if list_match:
                    units.extend(
                        self._flush_paragraph(
                            paragraph_lines=paragraph_lines,
                            page_number=page.page_number,
                            section_stack=section_stack,
                            section_counters=section_counters,
                        )
                    )
                    paragraph_lines.clear()

                    section_path = self._section_path(
                        section_stack
                    )

                    section_level = (
                        section_stack[-1][0]
                        if section_stack
                        else 0
                    )

                    section_index = section_counters.get(
                        section_path,
                        0,
                    )

                    section_counters[section_path] = (
                        section_index + 1
                    )

                    units.append(
                        StructuralUnit(
                            unit_type=UnitType.LIST,
                            content=list_match.group(1),
                            page_numbers=[page.page_number],
                            section_path=section_path,
                            section_level=section_level,
                            section_index=section_index,
                        )
                    )

                    continue

                paragraph_lines.append(stripped)

            units.extend(
                self._flush_paragraph(
                    paragraph_lines=paragraph_lines,
                    page_number=page.page_number,
                    section_stack=section_stack,
                    section_counters=section_counters,
                )
            )

        return units

    @staticmethod
    def _flush_paragraph(
        paragraph_lines: list[str],
        page_number: int,
        section_stack: list[tuple[int, str]],
        section_counters: dict[str, int],
    ) -> list[StructuralUnit]:
        """Convert accumulated paragraph lines into a structural unit."""
        if not paragraph_lines:
            return []

        content = " ".join(
            line.strip()
            for line in paragraph_lines
            if line.strip()
        ).strip()

        if not content:
            return []

        section_path = StructureExtractor._section_path(
            section_stack
        )

        section_level = (
            section_stack[-1][0]
            if section_stack
            else 0
        )

        section_index = section_counters.get(
            section_path,
            0,
        )

        section_counters[section_path] = (
            section_index + 1
        )

        return [
            StructuralUnit(
                unit_type=UnitType.PARAGRAPH,
                content=content,
                page_numbers=[page_number],
                section_path=section_path,
                section_level=section_level,
                section_index=section_index,
            )
        ]

    @staticmethod
    def _section_path(
        section_stack: list[tuple[int, str]],
    ) -> str:
        """Build a hierarchical section path."""
        return " > ".join(
            title
            for _, title in section_stack
        )