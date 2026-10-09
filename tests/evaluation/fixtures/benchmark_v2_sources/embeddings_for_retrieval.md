# Embedding Models for Semantic Retrieval

## Introduction

A retrieval system that searches by meaning rather than by exact wording depends entirely on the quality of the vectors produced by its embedding model, because every downstream stage, from HNSW traversal to cross-encoder reranking, only ever sees numbers derived from that first encoding step.

If the embedding model cannot place two differently worded but semantically equivalent passages near each other in vector space, no amount of clever indexing or reranking afterward can recover the relevance that was lost at encoding time, which is why embedding model selection is treated as a foundational decision rather than an interchangeable implementation detail.

This document covers how bi-encoder embedding models are trained, the trade-offs between embedding dimensionality and retrieval quality, the distance metrics used to compare vectors, strategies for adapting a general-purpose embedding model to a specialized domain, and the operational failure modes that show up once an embedding model is running against real production traffic rather than a clean benchmark set.

## Bi-Encoder Architecture and Training

Most production retrieval embedding models, including the all-MiniLM-L6-v2 model used in this system, are bi-encoders: a single shared transformer encodes the query and the candidate passage independently into fixed-length vectors, and relevance is approximated afterward as the geometric distance between those two vectors.

This independence is what makes bi-encoders fast enough for large-scale retrieval, because every chunk in the corpus can be embedded once, offline, and stored, so that a query at inference time only requires one additional encoding pass plus a nearest-neighbor lookup rather than a forward pass over every document in the corpus.

The trade-off is that a bi-encoder never lets the query and the passage attend to each other directly during encoding, so subtle interactions between specific query terms and specific passage terms are necessarily compressed into two separate fixed-length vectors before any comparison happens, which is the central limitation that cross-encoder reranking exists to address in a later pipeline stage.

Bi-encoders intended for retrieval are typically trained with a contrastive objective: given a query and a known relevant passage, the model is pushed to pull their embeddings closer together while pushing the query's embedding away from embeddings of unrelated or weakly related passages drawn from the same training batch, commonly referred to as in-batch negatives.

The quality of the negative examples used during this contrastive training has an outsized effect on the resulting model's discriminative power, because negatives that are too easy teach the model very little beyond gross topic separation, while negatives that are deliberately hard, meaning passages that share vocabulary or topic with the query but do not actually answer it, force the model to learn finer semantic distinctions that transfer well to real retrieval traffic.

Some training pipelines additionally use a triplet loss formulation, explicitly presenting the model with an anchor query, a positive passage, and a hard negative passage in every training example, rather than relying solely on whatever negatives happen to fall into the same batch, and this explicit triplet structure tends to produce embeddings that are more robust to paraphrase variation at query time.

## Embedding Dimensionality and the Size-Quality-Latency Trade-off

Embedding dimensionality determines both how much semantic information a vector can represent and how expensive every downstream operation involving that vector becomes, from the storage footprint in PostgreSQL to the number of floating point comparisons performed during HNSW graph traversal.

The production system in this document uses sentence-transformers/all-MiniLM-L6-v2, which produces 384-dimensional embeddings, a deliberately compact representation chosen to keep encoding latency low and to keep the vector index small enough to traverse quickly, at some cost in absolute retrieval quality compared to larger models.

Internal micro-benchmarking on a 50,000-chunk technical documentation corpus compared all-MiniLM-L6-v2 against three larger alternatives, measuring mean single-query encoding latency at a batch size of 32 and relative Recall@10 against the same evaluation query set, with all-MiniLM-L6-v2 used as the 1.00 baseline for relative recall:

| Model | Embedding dimensions | Max input tokens | Mean encode latency (ms, batch=32) | Relative Recall@10 |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 384 | 256 | 6.2 ms | 1.00 (baseline) |
| all-mpnet-base-v2 | 768 | 384 | 15.8 ms | 1.09 |
| bge-base-en-v1.5 | 768 | 512 | 14.1 ms | 1.12 |
| bge-large-en-v1.5 | 1024 | 512 | 27.4 ms | 1.15 |

The relative recall gains from doubling or tripling embedding dimensionality are real but modest compared to the latency cost, which is why a 384-dimension model remains attractive for a production system that also budgets latency for a downstream cross-encoder reranking pass on every query, since the reranker, not the embedding model, dominates end-to-end latency in most configurations.

Doubling the embedding dimensionality does not just slow down the encoder itself; it also roughly doubles the per-vector storage cost in the chunk embeddings table and increases the number of floating-point operations performed at every HNSW graph hop during vector search, so the dimensionality decision compounds across the entire retrieval pipeline rather than being isolated to the encoding step alone.

A useful way to frame the decision is that embedding dimensionality mostly trades a small, fairly predictable recall improvement against a latency and storage cost that scales with every single query and every single stored chunk, whereas a cross-encoder reranking pass trades a larger latency cost for a much larger quality improvement on the specific small set of candidates it examines, which is one reason production systems often prefer a smaller embedding model paired with reranking over a single larger embedding model used alone.

## Distance Metrics for Comparing Embeddings

Once two pieces of text have been embedded into vectors, retrieval still requires a way to turn those two vectors into a single relevance score, and the three distance metrics used in practice are cosine similarity, dot product (inner product), and Euclidean (L2) distance, each of which behaves differently depending on whether embedding vectors are normalized to unit length.

Cosine similarity measures the angle between two vectors while ignoring their magnitude entirely, which makes it a natural fit for embedding models trained with a contrastive objective, because such training typically optimizes for the direction of the embedding rather than its length, and a passage that happens to be longer or shorter should not automatically receive a higher or lower score purely as an artifact of vector magnitude.

Dot product similarity, by contrast, is sensitive to vector magnitude, so if an embedding model's training objective causes longer or more topically dense passages to naturally produce longer vectors, dot product search will have a systematic bias toward those passages regardless of whether they are actually more relevant, unless the vectors are first normalized to unit length, at which point dot product and cosine similarity become mathematically equivalent.

Because all-MiniLM-L6-v2 embeddings are normalized to unit length before being stored, cosine distance and dot product search would, in principle, rank candidates identically for this corpus, and the production system standardizes on cosine distance for vector search specifically because it remains correct even if a future embedding model substitution does not happen to produce unit-normalized vectors.

Euclidean distance measures the straight-line distance between two vectors in the embedding space, and for unit-normalized vectors it is a monotonic transformation of cosine similarity, meaning the two metrics will always agree on the relative ranking of candidates even though the raw numeric values differ, so the practical choice between cosine distance and Euclidean distance for normalized embeddings is more about convention and tooling support than about any difference in retrieval outcome.

Embedding normalization is not optional bookkeeping; a model whose output vectors are not normalized before storage will silently produce a distance metric that conflates "semantically similar" with "similarly verbose," and this failure mode is especially dangerous because it does not produce an error, it simply produces subtly wrong rankings that look plausible in casual inspection and only reveal themselves as a systematic quality regression once measured against a graded evaluation set.

## Domain Adaptation and Fine-Tuning

A general-purpose embedding model trained on broad web and question-answering data will place generic paraphrases near each other reasonably well, but it has no particular reason to understand that, within a specific technical corpus, two differently named configuration parameters actually refer to the same underlying mechanism, or that a given acronym has a narrow, corpus-specific meaning that differs from its common usage elsewhere.

Domain adaptation addresses this gap by continuing to train, or fully fine-tuning, an embedding model on query-passage pairs drawn from the target domain, so that the vector space reorganizes around the distinctions that actually matter for that corpus rather than the distinctions that happened to matter for the model's original training data.

One practical approach to generating domain-specific training pairs without manual annotation is synthetic query generation: an instruction-following language model is prompted to read a passage from the target corpus and produce a plausible user question that passage would answer, and the resulting (synthetic query, real passage) pairs are then used exactly like human-labeled contrastive training pairs.

Synthetic query generation is attractive because it scales to an entire corpus without requiring human annotators, but it inherits whatever blind spots the generating language model has; a generator that tends to produce questions using the same vocabulary already present in the passage will teach the embedding model very little beyond what it already knew, whereas a generator explicitly prompted to paraphrase and use alternative terminology will produce more useful contrastive signal.

Domain adaptation introduces an operational obligation that general-purpose models do not have: whenever the fine-tuned model is updated, every previously embedded chunk in the corpus must be re-embedded with the new model weights, because vectors produced by two different versions of a model, even versions that are closely related through fine-tuning, are not guaranteed to be comparable in the same distance space, and mixing embeddings from two model versions inside a single HNSW index silently corrupts the index's distance assumptions.

## Multilingual and Cross-Lingual Considerations

A monolingual embedding model, including all-MiniLM-L6-v2 in its base configuration, is trained predominantly on English text, and while it will still produce some embedding for non-English input, there is no guarantee that semantically equivalent sentences in two different languages land near each other in the resulting vector space.

Purpose-built multilingual embedding models are trained explicitly with parallel or comparable corpora across many languages so that a sentence and its translation are pulled close together during training, which is a meaningfully different training objective than simply training a single model on a mixture of monolingual corpora in several languages without any explicit cross-lingual alignment signal.

A corpus that is predominantly English but occasionally contains embedded non-English terms, code identifiers, or proper nouns generally retrieves acceptably with a monolingual model, because the surrounding English context still carries most of the semantic signal, but a corpus that genuinely requires cross-lingual retrieval, where a query in one language must retrieve a passage written in another language, requires a model explicitly trained for that cross-lingual alignment, and substituting a monolingual model in that setting produces retrieval that looks reasonable for same-language queries while silently failing for cross-language ones.

## Embedding Drift and Model Versioning

Embedding drift refers to the gradual or sudden change in what a given input maps to in vector space, and it has two distinct causes that are worth separating: drift caused by swapping the embedding model itself, and drift caused by the underlying corpus content changing in ways the original model's training distribution did not anticipate.

Model-swap drift is the more dramatic and more obviously dangerous case, because two different embedding model checkpoints, even two checkpoints of the same architecture at different fine-tuning stages, do not share a common vector space; a query embedded with model version two and compared against chunks embedded with model version one will produce distances that are not meaningfully interpretable, even though the search will still execute without any error and return some ranked list of results.

For this reason, any embedding model upgrade in production must be treated as a full corpus re-embedding event rather than an incremental update: every previously stored chunk vector must be recomputed with the new model before the new model is allowed to serve live queries against that corpus, and the old and new indexes should not be queried interchangeably during the transition.

Content-distribution drift is subtler: if a corpus that previously consisted mostly of narrative documentation gradually accumulates heavily tabular, numeric, or code-heavy content that the embedding model was never well-trained on, retrieval quality for that newer content can degrade even though the embedding model itself never changed, and this kind of drift is much harder to detect because there is no single discrete event, like a model deployment, to correlate the quality change against.

## Known Failure Modes in Production

Short-query, long-document asymmetry is one of the most consistent failure modes observed with bi-encoder retrieval: a user's query is often a handful of words, while a candidate chunk can be several hundred tokens of dense technical prose, and because both are compressed into a fixed-length vector of the same dimensionality, a short query's vector tends to be a much coarser semantic summary than a long chunk's vector, which can cause the embedding model to overweight the chunk's dominant topic at the expense of a specific detail buried in the middle of that chunk that is actually what the query is asking about.

Out-of-vocabulary jargon is a second recurring failure mode: a corpus-specific term, acronym, or product name that the embedding model never encountered during its original training has no learned representation, and the model falls back to encoding it as something close to an unknown or rare subword sequence, which means two passages that both use that same unfamiliar term may not actually end up close together in vector space purely because the term itself carries no learned meaning, even though a human reader would immediately recognize both passages as being on the same topic.

Embedding collapse describes a failure mode where a poorly fine-tuned or over-trained model starts mapping a wide variety of distinct inputs to nearly the same region of vector space, which manifests in retrieval as an unusually large number of candidates all receiving almost identical, uninformatively high similarity scores regardless of actual topical relevance; this is typically caused by a contrastive fine-tuning run with negatives that were too easy, too few, or too repetitive, and it is one of the reasons hard-negative mining is treated as important rather than optional during domain adaptation.

Truncation silently discards information: all-MiniLM-L6-v2 has a maximum input length, and any chunk content beyond that length is truncated before encoding rather than raising an error, so a chunk that happens to contain its most relevant sentence near the end, past the truncation boundary, will be embedded as if that sentence were never present, which is one of the operational reasons chunk sizes are kept well under the embedding model's maximum sequence length rather than pushed up against it.

## Practical Production Considerations

Batching queries and chunks together for encoding rather than encoding one item at a time substantially improves throughput on the hardware actually used in production, because transformer encoders benefit from the parallelism of processing multiple sequences in a single forward pass, and most of the latency numbers reported earlier in this document assume batched encoding rather than single-item encoding.

Caching query embeddings for frequently repeated or near-identical queries avoids redundant encoding work, but caching should be scoped carefully to the current embedding model version, since a cache entry computed under a previous model version must never be served once that model version has been retired, for the same reason that mixed-version vectors cannot be compared meaningfully inside a single index.

Because every chunk in the corpus must be embedded exactly once at ingestion time and then reused for every future query, the one-time cost of embedding a large corpus is amortized across an effectively unlimited number of future queries, which is the structural reason bi-encoder retrieval scales so much better than any architecture that would require comparing a query against every document at query time.

## Evaluating Embedding Quality Independently of the Full Pipeline

Because the embedding model sits underneath vector search, reranking, and query decomposition, a degradation introduced purely at the embedding layer can be masked by improvements elsewhere in the pipeline, which is why embedding quality is also measured in isolation rather than only through end-to-end retrieval metrics.

One isolated measurement is embedding-space clustering purity: a held-out set of passages with known topical labels is embedded, clustered using the raw vectors, and the resulting clusters are compared against the known labels, giving a signal about whether the embedding space is organizing content along meaningful semantic lines before reranking ever gets a chance to compensate for a poor arrangement.

A second isolated measurement is paraphrase robustness: a fixed set of queries is rewritten by hand into several paraphrases that preserve the original intent, each paraphrase is embedded separately, and the cosine similarity between the original query's embedding and each paraphrase's embedding is recorded, with a well-behaved embedding model expected to keep that similarity high even when surface wording changes substantially.

These isolated checks matter operationally because they can be run cheaply on a small evaluation set without needing a fully ingested corpus or a live reranker, which makes them suitable as a fast regression check whenever the embedding model, its preprocessing, or its tokenizer configuration changes, well before a full retrieval benchmark run is needed to confirm the change is safe.

## Interaction Between Embedding Choice and Chunk Design

Embedding quality and chunk design are not independent decisions, because the embedding model only ever sees whatever text ends up inside a single chunk, so a chunk boundary that separates a claim from the qualifying detail that makes it correct will cause the embedding model to encode an incomplete thought regardless of how good the model itself is.

Very short chunks tend to produce embeddings that are dominated by whichever few distinctive terms happen to appear in that short span, which can make retrieval unexpectedly sensitive to exact keyword overlap even though the underlying model is a semantic, non-lexical encoder, simply because there is so little surrounding context to dilute the influence of any single term.

Very long chunks, on the other hand, risk the short-query, long-document asymmetry described earlier being compounded further, since a single fixed-length vector is now standing in for a wider span of distinct sub-topics, which tends to push the embedding toward the chunk's dominant theme and away from secondary details that a query might specifically be asking about.

This is one of the reasons the production chunking pipeline targets roughly five hundred tokens per chunk rather than leaving chunk size unconstrained: it is close to the largest span the embedding model can summarize into a single vector without that vector becoming an unhelpfully diffuse average of unrelated sub-topics, while still being large enough to avoid the keyword-sensitivity problem that very short chunks introduce.

## Instruction-Tuned and Asymmetric Retrieval Embeddings

Some embedding models are trained with an explicit asymmetry between how a query is encoded and how a passage is encoded, typically by prepending a short instruction string, such as a literal "query:" or "passage:" marker, before the text is passed through the shared encoder, which lets the model learn separate behavior for the two roles even though the underlying transformer weights are shared between them.

This asymmetric design exists because a query and a passage are not actually the same kind of text in practice: a query is typically short, underspecified, and phrased as a question or a fragment, while a passage is typically longer, self-contained, and phrased as a statement, and a model that is told explicitly which role it is encoding at any given moment can learn to compensate for this structural difference rather than being forced to treat both kinds of text identically.

all-MiniLM-L6-v2, the embedding model used in this production system, is a symmetric model: it has no instruction-prefix convention and encodes a query and a passage through exactly the same code path with no role indicator distinguishing one from the other, which is simpler to operate, since there is no prefix convention to get wrong or to apply inconsistently between ingestion-time chunk embedding and query-time embedding, but it also means the model cannot learn any specialized query-side or passage-side behavior the way an asymmetric model can.

Substituting a symmetric embedding model for an asymmetric, instruction-tuned one, or vice versa, is not a drop-in replacement: an asymmetric model's passage embeddings were computed with the passage-side instruction prefix applied, so any new query embedded without the corresponding query-side prefix would be compared against those passage vectors inconsistently with how the model was actually trained, which is an additional, model-specific correctness requirement that a symmetric model like all-MiniLM-L6-v2 does not impose on this pipeline.

## Negative Mining Strategies in Contrastive Training

The quality of a contrastively trained embedding model depends heavily on how its negative examples, meaning the passages the model is explicitly taught are not relevant to a given query, are selected during training, since negatives that are trivially easy to distinguish from the true positive teach the model very little beyond the coarsest possible topic separation.

Random in-batch negatives, meaning simply treating every other passage that happens to be in the same training batch as a negative for a given query, are the cheapest negative source to construct, since they require no additional retrieval or ranking step beyond however the training batches were already assembled, but they tend to be easy negatives in aggregate, since a randomly sampled passage is statistically unlikely to share much topical overlap with any specific query at all.

Hard negative mining instead deliberately selects negatives that are topically close to the true positive, for example by first running an existing retrieval system, whether a simpler keyword-based system or an earlier version of the embedding model itself, against the training query and treating its highest-ranked incorrect results as hard negatives, which forces the model being trained to learn finer distinctions than topic alone, since the hard negatives already share substantial surface-level or topical similarity with the actual answer.

Hard negative mining introduces a specific risk called false negatives: a passage retrieved as a plausible hard negative candidate may, on closer inspection, actually also be a valid answer to the query that was simply not labeled as such in the training data, and training a model to actively push away from a passage that is, in truth, relevant teaches it precisely the wrong lesson for that example, which is why hard negative mining pipelines typically include some filtering step intended to catch and discard likely false negatives before they are used in training, even though no such filtering step can catch every case with certainty.

## Query-Side Truncation at Inference Time

The same maximum input length that governs chunk truncation at ingestion time also applies symmetrically to queries at retrieval time: all-MiniLM-L6-v2 accepts at most 256 input tokens, and a query submitted with more tokens than that is truncated before encoding rather than rejected, which means an unusually long, highly detailed user query can have its final portion silently dropped from what the embedding model actually sees, exactly analogous to the chunk-truncation risk described earlier but occurring on the query side of the comparison instead.

In practice this is a smaller concern for queries than for chunks, since most user queries, even unusually long or multi-part ones, still fall well under 256 tokens, whereas chunk content is deliberately allowed to approach roughly five hundred tokens under the production chunking budget, meaning chunks are considerably more likely to approach or exceed the embedding model's input ceiling than queries are; query-side truncation becomes a practical concern mainly for unusually long, densely multi-part queries that a user has phrased as a single very long sentence rather than as several shorter ones.

## Limitations

The embedding model documented here is a single fixed bi-encoder; the system has no mechanism to dynamically route a query to a different embedding model based on query type, length, or detected language, so every query and every chunk, regardless of content, is encoded with the same 384-dimension all-MiniLM-L6-v2 model, and any of the asymmetries described above apply uniformly across the whole corpus.

Because retrieval quality is bounded by whatever the embedding model actually captured at encoding time, no later pipeline stage, including cross-encoder reranking, can introduce semantic distinctions that the embedding step discarded entirely; reranking can only reorder candidates that vector search already surfaced, which makes embedding model quality a hard ceiling on overall system recall rather than something later stages can fully compensate for.
