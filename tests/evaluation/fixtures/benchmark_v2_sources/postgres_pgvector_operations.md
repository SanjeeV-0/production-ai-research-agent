# PostgreSQL and pgvector for Production Vector Search

## Introduction

Running vector search inside PostgreSQL rather than in a dedicated vector database is a deliberate architectural choice in this system: chunk content, chunk metadata, and chunk embeddings all live in the same relational database, queried through the pgvector extension, which avoids the operational overhead of keeping a second data store synchronized with the primary one at the cost of taking on PostgreSQL-specific index tuning and maintenance work that a purpose-built vector database might otherwise absorb.

This document covers how pgvector stores and indexes embeddings, the operational differences between the IVFFlat and HNSW index types it supports, the distance operators used for similarity search, schema design for chunk storage, and the maintenance and scaling considerations that come with running vector search as part of an otherwise ordinary relational database rather than as a separate specialized system.

## The pgvector Extension and Vector Storage

pgvector adds a native `vector` column type to PostgreSQL, storing a fixed-length array of floating point values directly alongside a row's other columns, so a table holding chunk content can store that chunk's 384-dimensional embedding in the same row rather than requiring a join or a lookup into a separate system to associate content with its vector representation.

Because the vector type has a fixed dimensionality declared at column-creation time, every embedding stored in a given column must share that same dimensionality, which means an embedding model upgrade that changes output dimensionality, for example moving from a 384-dimension model to a 768-dimension one, requires either a new column or a new table rather than simply overwriting the existing column's values in place.

Without any index at all, pgvector can still perform similarity search by sequentially scanning every row and computing the configured distance metric against the query vector, which is exact, meaning it is guaranteed to find the true nearest neighbors with no approximation error, but scales linearly with the number of stored vectors, making sequential scan impractical once a corpus grows into the range of hundreds of thousands or millions of chunks.

## Index Types: IVFFlat and HNSW

IVFFlat partitions the vector space into a configured number of clusters during index build, using k-means-style clustering, and at query time only searches within the small number of clusters nearest to the query vector rather than scanning the entire table, which is fast to build and keeps memory overhead low, but the clusters are fixed at build time, so as new vectors are inserted into an existing IVFFlat index, the cluster boundaries no longer necessarily reflect the true shape of the updated data distribution until the index is rebuilt.

HNSW builds a multi-layer navigable graph structure where each vector is a node connected to its approximate nearest neighbors, and search proceeds by greedily traversing the graph from an entry point toward the query vector's neighborhood, which generally achieves higher recall than IVFFlat at a comparable query latency, at the cost of a substantially more expensive index build process and a larger memory footprint, because the graph structure itself, not just the raw vectors, must be held in memory for efficient traversal.

An internal comparison of the two index types on a one-million-vector corpus of 384-dimension embeddings, using default-range build parameters for each, produced the following illustrative figures:

| Index type | Build time (1M vectors) | Memory overhead vs. raw vectors | Typical Recall@10 | Behavior under incremental insert |
| --- | --- | --- | --- | --- |
| IVFFlat | ~4 minutes | ~1.1x | 0.85-0.92, depends on probe count | Degrades gradually until reindexed |
| HNSW | ~22 minutes | ~1.6-2.0x | 0.93-0.97 | Graph updated in place, degrades much more slowly |
| Sequential scan (no index) | N/A | None | 1.00 (exact) | Unaffected by inserts |

The production system uses an HNSW index specifically because its recall advantage and its graceful behavior under incremental inserts outweigh the higher build time and memory cost, given that the corpus receives ongoing document ingestion rather than being built once and left static, which is exactly the workload pattern where IVFFlat's need for periodic full rebuilds becomes operationally inconvenient.

HNSW's recall is governed primarily by two build-time parameters, commonly named `m`, which controls how many graph connections each node maintains, and `ef_construction`, which controls how thoroughly the graph is explored while inserting each new node during the build; higher values of both parameters generally improve recall at the cost of longer build time and larger memory footprint, and these parameters cannot be changed for an existing index without rebuilding it from scratch.

## Search-Time Behavior and ef_search

Independent of the build-time parameters, HNSW search accepts a query-time parameter, commonly called `ef_search`, that controls how many candidate nodes the graph traversal considers before returning results; increasing `ef_search` widens the search breadth at each step of the traversal, which tends to improve recall because the traversal is less likely to prematurely settle on a locally good but globally suboptimal neighborhood, at the direct cost of additional query latency, since a wider search examines more graph nodes per query.

Unlike the build-time parameters, `ef_search` can be adjusted per query without rebuilding the index, which makes it the primary lever available for trading query latency against recall once an HNSW index is already built and in production, and it is the parameter most directly comparable to the `candidate_limit` setting used elsewhere in the retrieval pipeline, since both control how much of the index is examined before results are returned to the next pipeline stage.

Because ef_search operates purely within the approximate graph traversal, raising it can never recover a vector that the graph structure itself failed to connect well during index build; a poorly connected region of the graph, caused by build-time parameters that were too low for the corpus's actual distribution, will continue to produce weak recall in that region regardless of how high ef_search is set at query time, which is why build-time and query-time tuning are treated as separate, complementary levers rather than substitutes for one another.

## Distance Operators

pgvector exposes three distance operators corresponding to the three distance metrics discussed elsewhere in this corpus: `<->` computes Euclidean (L2) distance, `<#>` computes the negative inner product, and `<=>` computes cosine distance, and an index must be built specifying which of these operators it supports, since the graph or cluster structure built for one distance metric does not transparently serve queries issued under a different metric.

The production schema builds its HNSW index using the cosine distance operator `<=>`, consistent with the embedding model's unit-normalized output vectors, and every vector search query issued by the retrieval service uses that same operator, since issuing a query with a different operator than the one the index was built for would either fail to use the index at all or, if PostgreSQL silently falls back to a sequential scan, would still return correct results but without any of the performance benefit the index exists to provide.

Because pgvector returns a distance rather than a similarity score, with smaller values indicating closer vectors under the `<=>` operator, candidate chunks are ordered by ascending distance, and the merge-by-chunk-id deduplication logic used when multiple subqueries from query decomposition return overlapping candidates retains the lowest distance value observed for a given chunk across all subqueries, treating distance, not similarity, as the quantity to minimize when deciding which observation to keep.

## Schema Design for Chunk Storage

Each chunk is stored as a row referencing its parent document by a foreign key, along with a chunk index recording its position within that document, the chunk's textual content, its section path and section level inherited from the structural extraction step, any table-specific metadata such as a table identifier and fragment index when the chunk originated from a table, and the embedding vector itself in a dedicated vector column.

Storing chunk content and chunk embeddings in the same row, rather than splitting them across two tables joined at query time, avoids an extra join on the hot path of every single retrieval query, which matters because vector search is already the most index-dependent operation in the pipeline, and adding a join against a second large table on every query would introduce additional query planning complexity without a corresponding benefit in this schema's access pattern.

Document identity is tracked separately from chunk identity through a logical document identifier that remains stable across re-ingestion of an updated version of the same source document, which allows a newer version of a document to be ingested as a new set of chunks while the system can still recognize that those chunks logically supersede an earlier version's chunks rather than treating them as an entirely unrelated document.

## Query Planner Behavior

PostgreSQL's query planner decides whether to use the HNSW index or fall back to a sequential scan based on its own cost estimates, and for a vector similarity query this decision is generally straightforward once an appropriate index exists and the query is expressed using the operator the index was built against, but the planner can still choose a sequential scan under certain conditions, such as when the table is small enough that scanning it directly is estimated to be cheaper than the overhead of graph traversal, or when the query combines the vector similarity condition with additional filtering conditions that the planner estimates will eliminate most candidate rows before the vector comparison would even matter.

A sequential scan fallback is not itself an error condition and still returns exact, correct nearest-neighbor results, but it reintroduces the linear scaling behavior that the HNSW index exists specifically to avoid, so a query that silently falls back to sequential scan on a large table will typically be noticeably slower than expected without producing any explicit failure, which makes query plan inspection a necessary diagnostic step whenever vector search latency is higher than anticipated.

## Scaling and Operational Considerations

Connection pooling matters more for a vector-search workload than it might for a typical lightweight relational query, because each connection used for an HNSW search can briefly hold a meaningful amount of working memory for graph traversal, so a connection pool sized without accounting for vector search's per-query memory profile can lead to memory pressure under concurrent query load even when the number of concurrent connections looks modest by ordinary relational standards.

Read replicas can serve vector search queries once their HNSW index has caught up with the primary through normal replication, but because HNSW index updates are applied as part of the same write path as any other row change, a replica that is lagging behind the primary will serve stale results, including results that do not yet reflect a currently in-progress document ingestion, which matters specifically for a system where new documents can be ingested at any time rather than the corpus being fixed.

Partitioning chunk storage by logical document, or by a reasonable grouping of documents, is a scaling option considered for very large corpora, since it can bound the size of any single index that needs to be traversed per query, but it also means a query that could plausibly match chunks across multiple partitions must either search each partition independently and merge results, which adds coordination overhead, or must accept that a single HNSW index spans the entire partitioned range, which reintroduces the original scaling problem partitioning was meant to address.

## Backup, Restore, and Index Rebuilding

An HNSW index is not typically included in a logical backup the same way table data is, because the graph structure is a derived data structure that can always be reconstructed from the underlying vectors; a restore procedure therefore generally needs to rebuild the HNSW index from scratch after the underlying chunk data and embeddings are restored, and this rebuild time scales with corpus size in roughly the same way the original index build time did.

Because HNSW rebuild after a restore can take a substantial amount of time on a large corpus, a restore procedure needs to account for a period during which either no vector search is available at all or vector search must temporarily fall back to sequential scan, and this window is a deliberate operational trade-off accepted in exchange for not needing to maintain and synchronize a separately backed-up index artifact.

## Failure Modes

Index staleness after bulk insert is one of the most common operational issues: a bulk ingestion job that inserts a very large number of new chunks in a short period can temporarily degrade HNSW recall for queries whose true nearest neighbors are among the newly inserted vectors, if those vectors have not yet been fully integrated into the graph's connection structure, which is distinct from but related to the broader index-lag monitoring concern of keeping track of the delay between a document becoming ingested and its chunks becoming reliably queryable.

Memory pressure during index build is a second recurring issue: because HNSW build time and memory overhead scale with both corpus size and the configured build parameters, an index build attempted on hardware sized only for the steady-state query workload, rather than for the heavier, transient resource demand of a build or rebuild, can exhaust available memory partway through the build process, which is one of the reasons index rebuilds are typically scheduled as planned maintenance operations with dedicated resources rather than triggered casually during normal operation.

Distance-operator mismatch is a subtler failure mode: if a query is ever issued using a distance operator different from the one the HNSW index was built against, whether due to a configuration error or a code path that was not updated consistently, the query will either silently fall back to a sequential scan, with a correct but much slower result, or in some configurations may not benefit from the index at all without any explicit warning, making this a failure mode that manifests purely as a latency regression rather than as an incorrect result.

## Iterative Scan and Filtered Vector Queries

A vector similarity query that is combined with an additional relational filter, such as restricting results to chunks belonging to a specific document or a specific section path, introduces a complication for approximate indexes like HNSW: the graph traversal is organized purely around vector distance, with no awareness of the additional filter condition, so a naive execution can traverse and reject many graph nodes that are close in vector space but fail the filter, before accumulating enough filter-passing results to satisfy the requested result count.

Iterative scanning addresses this by allowing the index traversal to progressively widen its search, examining more of the graph than a single unfiltered top-K query would need to, specifically when the initial traversal does not surface enough filter-passing candidates, which trades additional query latency for the ability to still return a correctly filtered result set rather than silently returning fewer results than requested.

A filtered query against a very selective condition, one that only a small fraction of the table's rows satisfy, is the case most likely to require this wider iterative traversal, since the graph's nearest-neighbor ordering by distance alone gives very little guidance about where the sparse filter-passing rows happen to sit within that ordering, whereas a broad or non-selective filter condition rarely requires the traversal to widen meaningfully beyond what an unfiltered query would already examine.

## Vacuum and Maintenance for Vector Tables

PostgreSQL's ordinary autovacuum process, which reclaims space from updated or deleted rows and keeps query planner statistics current, applies to a table containing a vector column the same way it applies to any other table, but a table that receives frequent updates to its embedding column, for example during a corpus-wide re-embedding event following an embedding model upgrade, can accumulate a larger volume of dead row versions than a comparably sized table of ordinary scalar columns, simply because each vector value is large relative to typical scalar column values.

An HNSW index does not automatically shrink or compact itself as the underlying rows it indexes are deleted or updated; deleted entries are logically removed from being returned by queries, but the graph structure's memory and storage footprint does not necessarily reflect those deletions until a maintenance operation, such as a full index rebuild, is performed, which means a table that has undergone substantial churn, many rows deleted or re-embedded, can carry a larger HNSW index footprint than its current live row count alone would suggest is necessary.

Routine maintenance for a vector-heavy table therefore typically includes periodically monitoring index bloat specifically, not just ordinary table bloat, and scheduling an index rebuild when that bloat becomes large enough to affect either query latency or memory footprint, which is a maintenance concern layered on top of, rather than replacing, the autovacuum behavior PostgreSQL already applies to the table's ordinary row storage.

## Connection-Level Query Timeouts

A statement timeout is configured on connections used for vector search specifically to bound how long any single query, including one that falls back to a sequential scan or one that triggers an unusually wide iterative scan against a highly selective filter, is allowed to run before PostgreSQL cancels it outright, which protects overall request latency from being dominated by one pathological query at the cost of that specific query failing outright rather than eventually completing very slowly.

This statement timeout is set generously relative to the typical latency figures described elsewhere in this corpus, specifically so that it only ever triggers for a genuinely anomalous query, such as one hitting an unexpected sequential-scan fallback on a large table or an iterative scan widened unusually far by an extremely selective filter condition, rather than triggering on an ordinary, merely somewhat-slower-than-typical HNSW search.

A query canceled by the statement timeout surfaces as an explicit database error to the retrieval service rather than as a silent empty result, which the service's documented reranker-timeout-style fallback behavior does not automatically cover, since this failure occurs earlier in the pipeline at the vector search stage itself rather than at the reranking stage; a vector-search-level statement timeout is treated as a request-level failure for that specific query rather than something the pipeline silently degrades around.

## Limitations

pgvector inside PostgreSQL trades some of the specialized performance ceiling of a purpose-built vector database for the operational simplicity of keeping vector search, chunk content, and chunk metadata inside a single already-operated database system; very large corpora, well beyond the scale this system currently targets, would eventually encounter index build times and memory requirements that a horizontally sharded, purpose-built vector database is better equipped to absorb.

There is no sparse or keyword-based retrieval mechanism layered alongside pgvector in this system, so any query whose correct answer depends primarily on an exact rare term, such as a specific error code or an uncommon identifier that the embedding model has no strong learned representation for, must still be found through semantic similarity alone, with no lexical fallback available to recover a match that dense vector search misses.
