from app.ingestion.semantic_shredder import shred_semantically
from app.ingestion.structure import StructuralUnit, UnitType


class FakeEmbeddingProvider:
    """Controlled embedding provider for semantic shredder tests."""

    def __init__(
        self,
        embeddings: list[list[float]],
    ) -> None:
        self.embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        assert len(texts) == len(self.embeddings)
        return self.embeddings


def _paragraph(
    content: str,
    section_path: str = "Results",
    index: int = 0,
) -> StructuralUnit:
    return StructuralUnit(
        unit_type=UnitType.PARAGRAPH,
        content=content,
        page_numbers=[1],
        section_path=section_path,
        section_level=1,
        section_index=index,
    )


def test_adjacent_similar_paragraphs_are_grouped() -> None:
    units = [
        _paragraph("Retrieval improves recall.", index=0),
        _paragraph("Improved recall increases accuracy.", index=1),
        _paragraph("The weather was sunny.", index=2),
    ]

    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],  # P1
            [1.0, 0.0],  # P2 -> similarity 1.0
            [0.0, 1.0],  # P3 -> similarity 0.0
        ]
    )

    result = shred_semantically(
        units,
        embedding_provider=provider,
        threshold=0.7,
    )

    assert len(result) == 2
    assert len(result[0].units) == 2
    assert len(result[1].units) == 1


def test_distant_similar_paragraphs_are_not_clustered() -> None:
    units = [
        _paragraph("Retrieval improves recall.", index=0),
        _paragraph("The weather was sunny.", index=1),
        _paragraph("Recall is important for search.", index=2),
    ]

    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],  # P1
            [0.0, 1.0],  # P2
            [1.0, 0.0],  # P3
        ]
    )

    result = shred_semantically(
        units,
        embedding_provider=provider,
        threshold=0.7,
    )

    assert len(result) == 3


def test_section_change_is_always_a_boundary() -> None:
    units = [
        _paragraph(
            "Introduction content.",
            section_path="Introduction",
            index=0,
        ),
        _paragraph(
            "More introduction.",
            section_path="Introduction",
            index=1,
        ),
        _paragraph(
            "Methodology content.",
            section_path="Methodology",
            index=2,
        ),
    ]

    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    result = shred_semantically(
        units,
        embedding_provider=provider,
        threshold=0.7,
    )

    assert len(result) == 2
    assert len(result[0].units) == 2
    assert result[1].section_path == "Methodology"


def test_table_is_atomic_boundary() -> None:
    units = [
        _paragraph(
            "The experiment produced these results.",
            index=0,
        ),
        StructuralUnit(
            unit_type=UnitType.TABLE,
            content="Table 1",
            page_numbers=[1],
            section_path="Results",
            section_level=1,
            section_index=1,
        ),
        _paragraph(
            "The table demonstrates improvement.",
            index=2,
        ),
    ]

    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],  # paragraph before table
            [1.0, 0.0],  # paragraph after table
        ]
    )

    result = shred_semantically(
        units,
        embedding_provider=provider,
        threshold=0.7,
    )

    assert len(result) == 3
    assert result[0].units[0].unit_type == UnitType.PARAGRAPH
    assert result[1].units[0].unit_type == UnitType.TABLE
    assert result[2].units[0].unit_type == UnitType.PARAGRAPH