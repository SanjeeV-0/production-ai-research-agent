"""Turns loader-extracted page text into an ordered list of `StructuralUnit`s
(HEADING/PARAGRAPH/LIST/TABLE), the first step of the active ingestion
pipeline (see app.ingestion.service). Regex-based, Markdown-aware: it
recognizes `#`-style headings, `-`/`*`/numbered list items, and GitHub-
Flavored-Markdown pipe tables, and tracks a heading stack to build each
unit's hierarchical `section_path`.

Table detection is intentionally limited to genuine Markdown pipe-table
syntax (a header row immediately followed by a `|---|---|`-style separator
row) -- this is a real, well-defined text format, not a fabricated
capability. `MarkdownLoader` passes a `.md` file's own table syntax straight
through to here unchanged; `DocxLoader` renders python-docx's genuinely
structured `document.tables` data as the SAME pipe-table syntax (via
`app.ingestion.table_serializer.table_to_markdown`) specifically so both
loaders' tables are detected by this one code path. `PDFLoader` has no
structural table extraction (`pypdf.extract_text()` returns flat text with
no layout information), so PDF tables are NOT detected -- they ingest as
ordinary (often garbled) paragraph text, which is a real, documented
limitation rather than something this extractor silently papers over.
"""

import re

from app.ingestion.loaders.base import LoadedPage
from app.ingestion.structure import StructuralUnit, TableData, UnitType
from app.ingestion.table_serializer import table_to_markdown

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

_LIST_PATTERN = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)$")

# A pipe-table row needs at least two cells (i.e. at least one internal `|`
# with non-empty content on both sides) -- a single bare `|` with nothing
# either side is not a meaningful table row.
_TABLE_ROW_PATTERN = re.compile(r"^\|?(.+\|.+?)\|?$")

# A GFM separator row: two or more dash-segments joined by `|`, each
# optionally `:`-anchored for alignment (e.g. `| --- | :---: |`). Requires
# at least one literal `|` so a plain Markdown horizontal rule (`---`) is
# never mistaken for a table separator.
_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$")


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
        # Reset per call (not per instance) so every table in THIS document
        # gets a distinct, stable table_id, regardless of whether a caller
        # reuses one StructureExtractor across multiple documents.
        self._table_counter = 0

        for page in pages:
            paragraph_lines: list[str] = []

            # Table detection needs one line of lookahead (the separator
            # row after a candidate header row) and then consumes a variable
            # number of further lines (the data rows), so this loop uses an
            # explicit index cursor rather than a plain `for line in ...`.
            lines = page.content.splitlines()
            index = 0

            while index < len(lines):
                stripped = lines[index].strip()

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
                    index += 1
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

                    while section_stack and section_stack[-1][0] >= level:
                        section_stack.pop()

                    section_stack.append((level, title))

                    units.append(
                        StructuralUnit(
                            unit_type=UnitType.HEADING,
                            content=title,
                            page_numbers=[page.page_number],
                            section_path=self._section_path(section_stack),
                            section_level=level,
                            section_index=0,
                        )
                    )

                    index += 1
                    continue

                table_match = (
                    _TABLE_ROW_PATTERN.match(stripped)
                    and index + 1 < len(lines)
                    and _TABLE_SEPARATOR_PATTERN.match(lines[index + 1].strip())
                )

                if table_match:
                    units.extend(
                        self._flush_paragraph(
                            paragraph_lines=paragraph_lines,
                            page_number=page.page_number,
                            section_stack=section_stack,
                            section_counters=section_counters,
                        )
                    )
                    paragraph_lines.clear()

                    table_unit, next_index = self._extract_table(
                        lines=lines,
                        start_index=index,
                        page_number=page.page_number,
                        section_stack=section_stack,
                        section_counters=section_counters,
                    )

                    units.append(table_unit)

                    index = next_index
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

                    section_path = self._section_path(section_stack)

                    section_level = section_stack[-1][0] if section_stack else 0

                    section_index = section_counters.get(
                        section_path,
                        0,
                    )

                    section_counters[section_path] = section_index + 1

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

                    index += 1
                    continue

                paragraph_lines.append(stripped)
                index += 1

            units.extend(
                self._flush_paragraph(
                    paragraph_lines=paragraph_lines,
                    page_number=page.page_number,
                    section_stack=section_stack,
                    section_counters=section_counters,
                )
            )

        return units

    def _extract_table(
        self,
        lines: list[str],
        start_index: int,
        page_number: int,
        section_stack: list[tuple[int, str]],
        section_counters: dict[str, int],
    ) -> tuple[StructuralUnit, int]:
        """Parse a Markdown pipe table starting at `lines[start_index]`.

        `lines[start_index]` is the header row and `lines[start_index + 1]`
        is already confirmed (by the caller) to be a valid separator row.
        Consumes every immediately-following line that still looks like a
        table row as a data row, stopping at the first blank/non-table line
        or end of page -- i.e. one contiguous block of pipe-table syntax
        becomes exactly one TABLE unit. Returns the unit and the index of
        the first line NOT consumed, so the caller's cursor can resume there.
        """

        header_cells = self._parse_table_row(lines[start_index])

        row_cells: list[list[str]] = []
        cursor = start_index + 2  # skip header row + separator row

        while cursor < len(lines):
            candidate = lines[cursor].strip()

            if not candidate or not _TABLE_ROW_PATTERN.match(candidate):
                break

            row_cells.append(self._parse_table_row(candidate))
            cursor += 1

        self._table_counter += 1

        table_data = TableData(
            table_id=f"table-{self._table_counter}",
            # Markdown pipe-table syntax has no standard title line -- never
            # fabricate one. A real title can only come from an explicit
            # heading, which already becomes its own HEADING unit/section.
            title=None,
            headers=header_cells,
            rows=row_cells,
            page_numbers=[page_number],
        )

        section_path = self._section_path(section_stack)
        section_level = section_stack[-1][0] if section_stack else 0
        section_index = section_counters.get(section_path, 0)
        section_counters[section_path] = section_index + 1

        unit = StructuralUnit(
            unit_type=UnitType.TABLE,
            # Re-serialized via the canonical table_serializer rather than
            # keeping the source lines verbatim, so a TABLE unit's `content`
            # always has the exact same formatting regardless of whether it
            # came from a hand-written .md file or DocxLoader's rendering.
            content=table_to_markdown(table_data),
            page_numbers=[page_number],
            section_path=section_path,
            section_level=section_level,
            section_index=section_index,
            table=table_data,
        )

        return unit, cursor

    @staticmethod
    def _parse_table_row(line: str) -> list[str]:
        """Split one pipe-table row into trimmed cell values."""
        trimmed = line.strip()

        if trimmed.startswith("|"):
            trimmed = trimmed[1:]

        if trimmed.endswith("|"):
            trimmed = trimmed[:-1]

        return [cell.strip() for cell in trimmed.split("|")]

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

        content = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()

        if not content:
            return []

        section_path = StructureExtractor._section_path(section_stack)

        section_level = section_stack[-1][0] if section_stack else 0

        section_index = section_counters.get(
            section_path,
            0,
        )

        section_counters[section_path] = section_index + 1

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
        return " > ".join(title for _, title in section_stack)
