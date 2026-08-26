from sqlalchemy import create_engine, text

from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

# ============================================================
# CHANGE THIS to your PostgreSQL connection string
# ============================================================
DATABASE_URL = "postgresql+psycopg://postgres:291076@localhost:5432/research_agent"


# ============================================================
# Generate embedding
# ============================================================
query_text = "retrieval augmented generation"

provider = SentenceTransformerEmbeddingProvider()
embedding = provider.embed_text(query_text)

print(f"Query: {query_text}")
print(f"Embedding dimensions: {len(embedding)}")

if len(embedding) != 384:
    raise ValueError(
        f"Expected 384 dimensions, got {len(embedding)}"
    )


# Convert Python list -> pgvector format
query_vector = "[" + ",".join(map(str, embedding)) + "]"


# ============================================================
# Vector similarity search
# ============================================================
sql = text("""
    SELECT
        id,
        content,
        embedding <=> CAST(:query_vector AS vector) AS cosine_distance
    FROM document_chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_vector AS vector)
    LIMIT 5;
""")


# ============================================================
# Execute
# ============================================================
engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    results = connection.execute(
        sql,
        {"query_vector": query_vector},
    ).fetchall()


# ============================================================
# Display results
# ============================================================
print("\nTop 5 similar document chunks:")
print("=" * 80)

for rank, row in enumerate(results, start=1):
    print(f"\n#{rank}")
    print(f"ID: {row.id}")
    print(f"Cosine distance: {row.cosine_distance}")
    print(f"Content:\n{row.content}")
    print("-" * 80)
