from uuid import uuid4

import pytest

from app.ingestion.section_map import SectionMap


def test_section_map_stores_and_retrieves_ids() -> None:
    section_map = SectionMap()
    section_id = uuid4()

    section_map.add(
        "Results",
        section_id,
    )

    assert section_map.get("Results") == section_id
    assert "Results" in section_map
    assert len(section_map) == 1


def test_duplicate_section_path_raises() -> None:
    section_map = SectionMap()

    section_map.add(
        "Results",
        uuid4(),
    )

    with pytest.raises(ValueError):
        section_map.add(
            "Results",
            uuid4(),
        )


def test_unknown_section_raises() -> None:
    section_map = SectionMap()

    with pytest.raises(KeyError):
        section_map.get("Unknown")