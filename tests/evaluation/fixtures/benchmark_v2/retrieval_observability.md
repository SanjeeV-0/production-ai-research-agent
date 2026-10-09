# Observability for Production Retrieval Systems

## Introduction

A retrieval pipeline that passes every offline benchmark can still degrade silently once it is running against real production traffic, because benchmarks are necessarily run against a fixed snapshot of the corpus and a fixed set of queries, while a live system faces an evolving corpus, evolving query patterns, and infrastructure dependencies, such as the embedding provider and the decomposition language model, that can degrade independently of anything the retrieval logic itself does.

This document describes how production retrieval systems are instrumented and monitored: per-stage latency tracking, embedding drift detection, index staleness monitoring, logging strategy, alerting thresholds, and the triage process used to distinguish which specific pipeline stage is responsible when overall retrieval quality or latency degrades.

## Per-Stage Latency Instrumentation

Measuring only end-to-end request latency is insufficient for diagnosing where a slowdown originates, since a regression in any one of embedding, vector search, merge and deduplication, or reranking can produce an identical-looking increase in total latency, which is why each stage is instrumented separately with its own latency measurement rather than relying on a single aggregate timer around the whole request.

Latency is tracked using percentiles rather than a simple average, specifically p50, the median latency representing typical request behavior, p95, representing the slower tail that a meaningful fraction of requests actually experience, and p99, representing the rare but occasionally severe worst case, because an average can look entirely healthy even while a significant fraction of requests are experiencing latency bad enough to matter, if those slow requests are a minority diluted by a much larger number of fast ones.

Because cross-encoder reranking is the dominant latency stage under normal operation, as established elsewhere in this corpus, its p95 and p99 figures are watched with particular attention, since a regression specifically in reranking latency, for example caused by a larger-than-expected candidate pool or a slower-than-expected reranker deployment, will dominate the end-to-end latency regression almost entirely on its own.

## Embedding Drift Detection

Embedding drift detection monitors whether the statistical distribution of newly computed query or chunk embeddings is changing over time relative to a reference distribution captured when the current embedding model was first deployed, which can reveal a content-distribution shift, such as a corpus gradually accumulating a different mix of content than it originally had, well before that shift becomes visible as an obvious drop in any single retrieval metric.

One practical drift signal is tracking the mean and variance of query embedding vector components over a rolling time window and comparing them against the reference distribution captured at deployment time, since a meaningful shift in these aggregate statistics, even without any single dramatic query, can indicate that the query distribution the embedding model is now seeing differs from the distribution it was originally validated against.

Drift detection is explicitly a leading indicator intended to prompt investigation, not a direct measurement of retrieval quality itself, since a detected shift in embedding distribution does not by itself prove that retrieval quality has actually degraded, only that the input distribution has changed in a way that warrants checking whether quality has been affected as a result.

## Index Staleness Monitoring

Index staleness refers to the delay between a document being ingested, or an existing document being updated, and that document's chunks becoming reliably present and well-integrated into the HNSW index for querying, and this lag matters operationally because a user who expects a just-updated document to be immediately reflected in retrieval results will otherwise observe stale or missing results during that lag window.

Index lag is measured as the elapsed time between a document's ingestion completion timestamp and the first point at which a targeted query for that document's known content reliably retrieves its chunks, and this measurement is distinct from the HNSW recall degradation that can occur immediately after a large bulk insert, since lag monitoring is concerned with availability and integration timing, while bulk-insert recall degradation is concerned with ranking quality once the chunks are already queryable.

A sudden, sustained increase in index lag, beyond what is expected for ordinary steady-state ingestion volume, is treated as an early signal of either an ingestion pipeline slowdown or a database-side issue affecting how quickly new vectors are integrated into the index, and is monitored separately from the retrieval service's own request-serving latency, since the two can degrade independently of one another.

## Logging Strategy

Query logs record the original user query, whether decomposition was triggered and what subqueries were generated if so, the configured `candidate_limit` and `final_limit` for that request, and the final set of returned chunk identifiers, which together provide enough information to reconstruct what the pipeline decided to do for any individual request after the fact, without needing to re-run the request live to understand its behavior.

Candidate logs, recorded at a lower, sampled rate given their larger volume, capture the full merged candidate pool before reranking, including each candidate's vector distance and which subquery or subqueries surfaced it, which is specifically useful for diagnosing whether a quality issue originated upstream, in vector search failing to surface relevant candidates at all, or downstream, in reranking failing to rank already-present relevant candidates highly.

Rerank score logs capture each candidate's score from the cross-encoder together with its final rank position, which allows a later analysis to check for signs of the calibration and distractor-promotion failure modes described elsewhere in this corpus, such as a pattern where candidates sharing exact query vocabulary are consistently scored higher than semantically equivalent candidates using different terminology.

Query logs retain the literal text of user queries, which can include sensitive or personally identifying information depending on what users ask, so logging retention policy and access controls for query logs specifically are treated as a distinct concern from the retention policy applied to candidate and rerank score logs, which contain corpus content rather than user-submitted text.

## Alerting Thresholds

| Metric | Normal range | Alert threshold | Severity |
| --- | --- | --- | --- |
| p95 end-to-end latency | 150-350 ms | sustained above 600 ms for 5 minutes | high |
| Reranker error rate | below 0.5% | above 2% over 10 minutes | high |
| Embedding provider timeout rate | below 0.1% | above 1% over 10 minutes | medium |
| Index lag (ingestion to queryable) | below 60 seconds | above 10 minutes | medium |
| Query decomposition LLM failure rate | below 1% | above 5% over 15 minutes | medium |

Alert severity in this table reflects the blast radius and user-facing impact of each failure rather than purely its underlying technical cause: a reranker error, for example, affects every single request that reaches the reranking stage and therefore carries high severity, while a decomposition failure affects only requests that would have triggered decomposition and, given the documented fallback of treating the original query as the only subquery, degrades those requests gracefully rather than failing them outright, which is why it is rated at medium severity despite also representing an external dependency failure.

Alert thresholds are deliberately set as sustained conditions over a window, rather than triggering on a single instantaneous bad data point, specifically to avoid paging on transient noise, such as one unusually slow individual request, while still reliably catching a genuine, ongoing degradation that would meaningfully affect a large share of real traffic if left unaddressed.

## Failure Triage Workflow

Distinguishing an embedding provider failure from an index failure from a reranker failure begins with per-stage latency and error-rate data, since each of these three failure sources tends to leave a distinct signature: an embedding provider failure typically manifests as an elevated error or timeout rate specifically on the embedding stage with vector search, merge, and reranking stages remaining unaffected because they are never reached for the failed requests.

An index-related failure, such as the query planner falling back to sequential scan due to a distance-operator mismatch or an index that failed to build correctly, typically manifests as a disproportionate increase in vector search stage latency specifically, while the embedding stage and the reranking stage, once reranking does receive a candidate pool, continue to behave normally, since the index issue is isolated to the search step itself.

A reranker failure or a reranker-side latency regression manifests most visibly in end-to-end p95 and p99 latency, given that reranking already dominates overall latency under normal operation, and can be further isolated by checking whether rerank latency has increased at a fixed candidate pool size, which would point to a reranker-side regression specifically, as opposed to rerank latency increasing only because candidate pool sizes have grown for an unrelated reason, such as a shift toward more decomposition-triggering queries.

A quality-only degradation, where latency remains normal across every stage but offline or online quality metrics decline, points away from any single stage outright failing and toward a content or distribution issue instead, such as embedding drift from a changing corpus, a decomposition language model that has grown less reliable, or a genuine shift in what kinds of queries users are submitting that the current chunking and embedding configuration handles less well than before.

## Canary Deployment for Model Upgrades

Any upgrade to the embedding model or the cross-encoder reranker is treated as a high-risk change given the full-corpus re-embedding requirement discussed elsewhere in this corpus, and such an upgrade is validated through a canary deployment, where the new model version serves a small, controlled fraction of live traffic while the previous version continues serving the remainder, with both versions' metrics compared directly against each other before a full rollout.

Canary validation for an embedding model upgrade specifically requires the canary's corpus to already be re-embedded under the new model version before the canary receives any live traffic, since serving a query embedded under a new model against an index still built from the old model's vectors would compare incomparable vector spaces, producing results that look superficially plausible but are not meaningfully ranked at all.

A reranker upgrade's canary validation is comparatively simpler operationally, since the reranker does not require any corresponding re-embedding of the corpus, only the new reranker model needs to be available to serve its fraction of canary traffic, which is one reason reranker upgrades are generally considered lower-risk operational changes than embedding model upgrades despite both being model-replacement events.

## Synthetic Canary Queries for Continuous Health Checking

Because a full offline benchmark is comparatively expensive to run and is therefore only executed periodically, such as after a deliberate configuration or model change, a smaller set of synthetic canary queries with known expected chunks is instead run continuously against the live production system on a fixed schedule, specifically to catch a quality regression that develops silently between scheduled full benchmark runs.

A synthetic canary query differs from a full benchmark query mainly in scale and purpose: rather than covering the full breadth of difficulty levels and categories a complete benchmark aims for, canary queries are chosen specifically because their expected chunk is well understood and highly stable, so that a canary failure, meaning the expected chunk unexpectedly drops out of the top results, is a strong and immediate signal of a genuine regression somewhere in the live pipeline rather than an ambiguous borderline case.

Canary query failures are treated as a high-priority signal precisely because they run continuously against the actual production system end to end, including real infrastructure dependencies such as the live embedding provider and the live reranker deployment, which lets them catch a class of regression that a benchmark run in a separate evaluation environment might not reproduce if that evaluation environment does not fully mirror production's live infrastructure and configuration.

## Cost and Resource Monitoring

Reranking is the most compute-intensive stage in the pipeline and is therefore also the primary driver of infrastructure cost per query, so GPU or CPU utilization on whatever hardware serves the cross-encoder reranker is tracked as a first-class operational metric alongside the latency and error-rate metrics described earlier, since a utilization trend climbing toward saturation is a leading indicator of an impending latency regression before that regression actually shows up in the p95 and p99 latency figures.

Cost per query, estimated from the combination of embedding compute, vector search compute, reranking compute, and the OpenRouter API cost incurred whenever decomposition is triggered, is tracked as a distinct metric from latency, since a configuration change that improves latency by, for example, reducing candidate pool size does not necessarily reduce cost proportionally, and conversely a change that increases decomposition frequency can increase cost noticeably through additional OpenRouter calls even if it has only a modest effect on measured end-to-end latency.

Monitoring the OpenRouter-related cost and call volume separately from the rest of the pipeline's infrastructure cost is particularly useful given that this is the one stage billed per external API call rather than drawn from the team's own provisioned compute capacity, which means a shift in query patterns toward more decomposition-triggering queries can increase this specific cost component without any change at all to the provisioned infrastructure capacity for the rest of the pipeline.

## Correlating Observability Signals With Benchmark Categories

Production query logs are tagged, where feasible, with an inferred category similar to the category labels used in offline benchmark construction, such as whether a query triggered decomposition, how large its resulting candidate pool was, and roughly how many chunks were ultimately returned as relevant by downstream engagement signals, which allows a live production quality concern to be connected back to a specific benchmark category rather than treated only as an undifferentiated aggregate quality question.

This correlation is particularly useful when an offline benchmark shows a specific category, for example cross-document or decomposition-benefiting queries, underperforming relative to simpler categories, since production logs tagged with comparable category signals can then be inspected to estimate what share of real live traffic actually falls into that weaker category, which helps prioritize whether addressing that category's weakness is worth the engineering effort relative to how often real users actually submit that kind of query.

Without this kind of correlation, an offline benchmark's category-level breakdown and live production monitoring would otherwise remain two entirely separate sources of signal, one reflecting a fixed, curated query set and the other reflecting live but uncategorized traffic, with no direct way to translate a benchmark weakness into an estimate of its actual real-world frequency or impact.

## Limitations

Observability in this system instruments and alerts on the pipeline's own internal stages and their latency and error behavior, but it does not itself include an automated, continuously running quality benchmark against live production traffic; detecting a genuine retrieval quality regression that does not also manifest as a latency or error-rate anomaly still depends on periodically rerunning an offline benchmark or on observing online signals such as user feedback, rather than being caught automatically by the alerting thresholds described in this document.

Index lag and embedding drift monitoring both depend on comparing current behavior against a reference baseline captured at some earlier point in time, which means both forms of monitoring are only as useful as how representative and how recently updated that reference baseline actually is; a reference baseline that is itself stale can cause drift detection to either miss a genuine, gradual shift that has been ongoing since before the baseline was last refreshed, or to flag a difference that no longer reflects the system's actual current steady state.
