# Reranking and Candidate Pool Design

## Introduction

Vector search is optimized to be fast across an entire corpus, which it achieves by comparing compact, independently computed embeddings rather than by directly comparing the full text of a query against the full text of every candidate, but that same compactness is also what limits how precisely it can judge relevance, since two pieces of text can be reasonably close in vector space without actually being specifically relevant to each other in the way the query intends.

Reranking exists to recover the precision that bi-encoder vector search necessarily gives up in exchange for speed, by taking the comparatively small candidate pool vector search already assembled and scoring each candidate against the query using a model that is allowed to look at the query and the candidate together, rather than as two separately pre-computed vectors.

This document describes how cross-encoder reranking works at a conceptual level, how it differs from pointwise, pairwise, and listwise ranking approaches more generally, how reranking latency scales with candidate pool size, and the specific ways reranking can fail even when it is functioning as designed.

## Cross-Encoder Scoring

A cross-encoder reranker takes the query and a single candidate chunk together as one combined input to a transformer model, allowing every token of the query to attend directly to every token of the candidate during encoding, which produces a single relevance score for that specific query-candidate pair rather than two independent vectors that are compared afterward.

This joint encoding is what gives a cross-encoder its precision advantage over a bi-encoder: because the model sees the query and candidate simultaneously, it can pick up on specific interactions, such as a query term that is negated, qualified, or contradicted by specific wording in the candidate, that two separately computed embeddings compressed into fixed-length vectors ahead of time have no mechanism to represent.

The cost of this precision is that a cross-encoder cannot be run once per chunk ahead of time the way a bi-encoder's embeddings can; a cross-encoder score is specific to one particular query, so scoring a candidate pool of a given size requires that many full forward passes through the cross-encoder model at query time, for every single query the system receives, which is the structural reason cross-encoder reranking is reserved for a small post-vector-search candidate pool rather than applied across an entire corpus directly.

## Pointwise, Pairwise, and Listwise Approaches

A pointwise reranker, which is the approach used in this system's cross-encoder, scores each query-candidate pair independently and produces one absolute relevance score per candidate, after which candidates are simply sorted by that score; this is the simplest approach to implement and reason about, since each candidate's score does not depend on any other candidate in the pool.

A pairwise reranker instead compares two candidates directly against each other at a time, producing a judgment about which of the pair is more relevant to the query rather than an absolute score for either one individually, and a full ranking is then derived by aggregating many such pairwise judgments, which can capture relative distinctions between similar candidates more precisely than two independent pointwise scores might, at the cost of needing many more model invocations to cover every pair in a candidate pool.

A listwise reranker considers the entire candidate pool at once and directly outputs an ordering or a set of scores calibrated relative to each other within that specific pool, which can in principle produce the most internally consistent ranking of the three approaches, since it never has to reconcile independently computed scores or pairwise judgments after the fact, but it requires an architecture capable of processing a variable-length set of candidates jointly, which is more complex to serve in production than a model that scores one query-candidate pair at a time.

The pointwise approach used in this system is chosen primarily for its operational simplicity and because each candidate's score can be computed independently, which makes the reranking stage straightforward to batch and parallelize across available compute, at the acknowledged cost of not capturing relative distinctions between two very similar candidates as explicitly as a pairwise or listwise approach might.

## Reranking Latency and Candidate Pool Size

Because a pointwise cross-encoder must score every candidate in the pool individually, reranking latency scales roughly linearly with candidate pool size, and this relationship is the central operational constraint shaping how large a `candidate_limit` a production deployment can realistically afford:

| candidate_limit | Mean rerank latency | Mean Recall@10 ceiling before rerank |
| --- | --- | --- |
| 20 | 32 ms | 0.47 |
| 50 | 78 ms | 0.61 |
| 100 | 145 ms | 0.69 |
| 200 | 290 ms | 0.74 |

This table, which mirrors the candidate-pool-sizing figures discussed from the architectural perspective elsewhere in this corpus, illustrates the core trade-off reranking introduces into candidate pool sizing: because reranking cannot improve a candidate's recall ceiling, only its ranking once it is already present, every increase in candidate pool size is paid for in rerank latency regardless of whether that larger pool actually contains any additional relevant chunk beyond what a smaller pool already included.

A useful way to think about this relationship is that candidate pool size sets an upper bound on recall, while reranking determines how well that bound is actually realized in the final top-K result; a very large candidate pool with a weak reranker can still produce a poor final result if relevant chunks are not ranked highly within that large pool, while a very small candidate pool with an excellent reranker is still capped by whatever recall ceiling that small pool allows.

## Input Truncation Behavior

A cross-encoder model has a maximum combined input length for the query and candidate together, and when a candidate chunk's content, combined with the query, exceeds that maximum, the input is truncated before scoring rather than rejected outright, which means a long chunk can have its tail end silently dropped from the reranker's view even though that same full chunk content was available in its entirety to the embedding model during vector search.

This creates a specific asymmetry worth being aware of: a chunk whose most relevant sentence happens to sit near the end of its content can be embedded correctly and placed well by vector search, since the embedding model saw the full chunk, but can then be scored poorly by the reranker if truncation caused the cross-encoder to never actually see that specific sentence, producing a case where a genuinely relevant candidate is demoted during reranking for reasons unrelated to its actual relevance.

Because chunk sizes in this pipeline are kept to a target of roughly five hundred tokens specifically to stay comfortably under both the embedding model's and the reranker's input limits, this truncation risk is reduced but not eliminated for chunks that occasionally exceed the target budget, such as those containing an indivisible structural unit like a large table fragment.

## Failure Modes

Distractor promotion describes a failure where the reranker assigns a high score to a candidate that shares substantial vocabulary or surface-level phrasing with the query, discusses a closely related mechanism, but does not actually answer what the query is asking; a cross-encoder's strength at picking up on fine-grained lexical and phrasal overlap can, in these cases, work against it, since lexical overlap with a plausible-sounding but incorrect candidate can resemble the pattern the model learned to associate with genuine relevance during training.

Over-confidence on lexical overlap is a related but narrower failure mode where the reranker consistently assigns high scores to candidates matching the query's exact terminology even when a more semantically appropriate candidate, phrased using different terminology for the same underlying concept, is also present in the pool, which tends to surface specifically for terminology-variation questions where the correct evidence uses a synonym or a related concept rather than the query's exact wording.

Calibration issues arise because a pointwise cross-encoder's absolute scores are not necessarily comparable in a meaningful way across different queries; a reranker can be well-calibrated for relative ordering within a single query's candidate pool while its absolute score values carry a different meaning from one query to the next, which matters for any downstream use of the reranker's raw scores, such as an absolute cutoff threshold, as opposed to relying purely on the relative ordering it produces.

The interaction between HNSW's recall ceiling and reranking is itself a structural failure mode worth naming explicitly: reranking can only ever reorder candidates that vector search actually surfaced into the candidate pool, so if HNSW search, whether due to the index's approximate nature or an insufficiently large `candidate_limit`, never places a genuinely relevant chunk inside the pool in the first place, no amount of reranking quality can recover that chunk, making vector search recall a hard upstream ceiling that reranking quality cannot compensate for no matter how well the reranker itself performs.

## Threshold Strategies Versus Fixed Top-K

A fixed top-K strategy, which is what this system uses, always returns exactly `final_limit` chunks regardless of how confidently the reranker scored them, which has the advantage of a predictable, constant-size result set but the disadvantage that a query with only one or two genuinely relevant chunks in its candidate pool will still have its result set padded out to `final_limit` with lower-relevance chunks that happened to score highest among the remaining candidates.

A score-threshold strategy would instead return only candidates whose reranked score exceeds some fixed cutoff, which can avoid padding a result set with weak candidates when few genuinely relevant ones exist, but given the calibration issue described above, where absolute score values are not necessarily comparable across different queries, a single fixed threshold tuned against one query distribution may behave inconsistently across a broader and more varied range of queries than it was tuned against.

This system relies on fixed top-K selection specifically because it sidesteps the cross-query calibration problem entirely, accepting the trade-off that some returned results will include lower-relevance chunks when a query's candidate pool simply does not contain `final_limit` worth of strongly relevant evidence.

## Candidate Pool and Final Limit Combinations

| candidate_limit | final_limit | Qualitative effect |
| --- | --- | --- |
| 20 | 3 | Fast but recall-limited; only suitable when evidence is expected to rank very highly already |
| 50 | 5 | Moderate balance; reasonable default for straightforward queries |
| 100 | 5 | Wider recall net with a tightly curated final result; favors precision in the returned set |
| 100 | 10 | Wider recall net with a looser final result; favors coverage over tight precision |
| 200 | 10 | Highest recall ceiling tested, at the highest reranking latency cost |

Choosing a specific combination from this table is a deployment-level decision rather than a property of the reranker itself, since the reranker behaves identically regardless of which combination of `candidate_limit` and `final_limit` it is given; what changes is purely how much material it is asked to evaluate and how much of its output is ultimately surfaced.

## Alternative Reranker Architectures Considered

Late-interaction architectures, of which ColBERT-style models are the best-known example, represent a middle ground between a bi-encoder and a cross-encoder: rather than compressing a passage into a single fixed-length vector or jointly encoding the query and passage together, a late-interaction model stores a separate embedding for every token of a passage and computes relevance at query time by matching each query token against its best-matching passage token, summing these token-level matches into a final score.

This approach can approach cross-encoder-level precision while remaining considerably cheaper at query time than a true cross-encoder, since the per-token passage embeddings can be precomputed once at ingestion time the same way bi-encoder embeddings are, with only the relatively cheap token-matching computation deferred to query time, rather than requiring a full transformer forward pass over the query and candidate together for every single candidate.

This production system uses a cross-encoder rather than a late-interaction reranker primarily for operational simplicity: a cross-encoder requires no additional per-token storage beyond the single chunk-level embedding already used for vector search, and no specialized token-matching infrastructure, whereas a late-interaction model would require storing and serving a substantially larger per-chunk representation, multiple embeddings per chunk rather than one, which was judged to add more operational complexity than its latency advantage over a cross-encoder justified at this system's current candidate pool sizes.

## Score Normalization and Downstream Consumption

Reranker scores are used internally purely to determine the relative ordering of candidates within a single request's candidate pool, and are not currently normalized onto any fixed, query-independent scale or exposed as an absolute confidence value to any downstream consumer of the retrieved chunks, consistent with the calibration limitation described earlier, where absolute score values are not reliably comparable across different queries.

Because only the ordering, not the raw score values, is treated as meaningful outside the reranking stage itself, the final top-K chunks are returned to downstream consumers as a ranked list without an accompanying confidence score field that any consumer might otherwise be tempted to threshold against, which avoids the failure mode of a downstream system applying a fixed absolute cutoff to a score that was never designed to be compared across queries in the first place.

## Reranker Fine-Tuning and Domain Adaptation

A general-purpose cross-encoder, trained on broad question-answering or passage-ranking data, can be fine-tuned further on query-passage relevance judgments specific to the target corpus, the same way a bi-encoder embedding model can be domain-adapted, with the goal of teaching the reranker which fine-grained distinctions actually matter for distinguishing genuinely relevant from superficially similar candidates within that specific corpus's terminology and conventions.

Reranker fine-tuning carries a distinct overfitting risk from embedding fine-tuning, because a cross-encoder has access to the full joint text of the query and candidate at scoring time, which means it can, if trained on a limited or unrepresentative set of in-domain examples, learn to key off incidental surface patterns specific to those training examples, such as a particular phrasing style or a specific set of terms that happened to co-occur with relevant passages in the training set, rather than a generalizable notion of relevance.

Because a fine-tuned cross-encoder, unlike a fine-tuned embedding model, does not require the existing corpus to be re-embedded when it is deployed, a reranker fine-tuning update is operationally lighter weight than an embedding model update of comparable scope, which is consistent with reranker upgrades generally being treated as lower-risk operational changes than embedding model upgrades.

## Limitations

Reranking in this system is always pointwise and always evaluates one candidate chunk independently of every other candidate in the same pool, so it has no mechanism to directly penalize near-duplicate candidates that say almost the same thing, or to explicitly reward diversity of evidence across the final result set; two highly similar chunks can both score highly and both be returned, displacing a third, more diverse but slightly lower-scoring chunk that might have added genuinely new evidence.

There is no feedback loop from reranking back into vector search within a single request; if the reranker determines that none of the candidates in the pool are strongly relevant, there is no mechanism to trigger an expanded or reformulated vector search within that same request, the pipeline simply returns the best-available reranked result from the original candidate pool it was given.
