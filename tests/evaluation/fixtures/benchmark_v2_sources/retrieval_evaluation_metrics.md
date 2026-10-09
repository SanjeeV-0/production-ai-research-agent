# Evaluation Metrics and Benchmark Design for Retrieval Systems

## Introduction

Measuring whether a retrieval system is actually good at its job requires more than reading a handful of example queries and judging the results by eye, because a system can look impressive on a few hand-picked examples while performing poorly across the broader distribution of real queries it will eventually face, and a system can equally look mediocre on a poorly constructed benchmark while actually performing well in practice.

This document describes the core offline evaluation metrics used to judge retrieval quality, how graded relevance judgments differ from simple binary relevance, the pitfalls that commonly compromise an offline benchmark's validity, how online evaluation signals complement offline benchmarks, and the practical considerations involved in collecting relevance judgments and interpreting the resulting metrics.

## Recall@K and Precision@K

Recall@K measures, out of all the chunks that are genuinely relevant to a query anywhere in the corpus, what fraction of them appear within the top K results actually returned, which makes it a coverage metric: a high Recall@K means the system is successfully surfacing most of the relevant material that exists, regardless of how that material is ordered within the top K or how much irrelevant material is mixed in alongside it.

Precision@K measures, out of the K results actually returned, what fraction of them are genuinely relevant, which makes it a purity metric: a high Precision@K means the returned results are mostly relevant, but it says nothing about whether additional relevant material exists elsewhere in the corpus that failed to make it into the top K at all.

These two metrics deliberately capture different failure modes and are not substitutes for one another: a system can achieve high recall by returning a very large K and thereby being very likely to include most relevant chunks somewhere in that large set, while simultaneously having low precision because the same large K also includes a great deal of irrelevant material, and conversely a system can achieve high precision with a very small K by returning only its single most confident result, while having low recall because many other relevant chunks elsewhere in the corpus are excluded entirely.

Because Recall@K and Precision@K move in different directions as K changes, with recall generally non-decreasing and precision generally non-increasing as K grows, neither metric alone adequately summarizes system quality, which is why production evaluation reports both together at the same K value, and why `candidate_limit` and `final_limit` are treated as a joint recall-versus-precision tuning decision rather than optimized against a single metric.

## Mean Reciprocal Rank and Hit Rate

Mean Reciprocal Rank, commonly abbreviated MRR, measures how early the first relevant result appears in the ranked list, computed per query as one divided by the rank position of the first relevant result, and then averaged across all queries in the evaluation set, which makes MRR especially sensitive to whether a system gets the single best answer near the very top of its results rather than merely somewhere within an acceptable range.

MRR's sensitivity to the first relevant result's position is also its principal limitation: a query with five equally relevant chunks scattered throughout the top ten results receives the exact same MRR contribution as a query with only one relevant chunk sitting at that same top position, since MRR only looks at the rank of the first relevant hit and is entirely blind to how many additional relevant results follow it or where they fall.

Hit Rate@K measures, per query, simply whether at least one relevant chunk appears anywhere within the top K results, expressed as a fraction across the full evaluation set, which makes it the least granular of the metrics described here: a query that returns exactly one relevant chunk at the very top of its results and a query that returns five relevant chunks spread across the top K both count identically as a single successful hit for that query under Hit Rate@K.

Hit Rate@K is still a useful complementary signal precisely because of this coarseness: it answers the narrow but practically important question of whether a system fails a query entirely, surfacing no relevant evidence anywhere in the top K, which is a meaningfully different and often more severe failure than surfacing some relevant evidence but ranking it suboptimally, a distinction that Recall@K and Precision@K do not directly isolate on their own.

## Graded Relevance and nDCG

Binary relevance judgments classify every chunk as either relevant or not relevant to a given query, with no distinction made between a chunk that directly and completely answers the query and a chunk that only provides tangential, incomplete context, which is a simplification that is easy to work with but discards information that matters for judging ranking quality specifically.

Graded relevance instead assigns each chunk a relevance level on a defined scale, commonly a four-point scale running from zero, meaning not relevant at all, through one, meaning related but insufficient context on its own, through two, meaning strongly supporting evidence, up to three, meaning the chunk directly answers or is essential evidence for the query, which allows a metric to distinguish a system that ranks a directly answering chunk first from a system that ranks only a tangentially related chunk first, even though a binary scheme would count both as equally successful hits.

Normalized Discounted Cumulative Gain, commonly abbreviated nDCG, is the standard metric for exploiting graded relevance: it sums each returned chunk's relevance grade, discounted by a factor that decreases with the chunk's rank position so that highly relevant chunks ranked near the top contribute more than equally relevant chunks ranked further down, and this sum is then normalized against the maximum possible score achievable by the ideal ordering of the same set of relevant chunks, producing a score that can be compared meaningfully across queries with different numbers of relevant chunks.

Using graded relevance and nDCG requires that relevance judgments themselves be collected on that same graded scale in the first place, which is a meaningfully larger annotation burden than binary judgments, since a human annotator, or a benchmark author constructing ground truth by hand, must make a finer distinction for every piece of evidence rather than a simple relevant-or-not call, which is one reason simpler binary-oriented metrics like Hit Rate@K remain common even in evaluation setups that have graded relevance data available.

## Offline Benchmark Construction Pitfalls

A benchmark corpus that is heavily template-based, where many passages share repeated boilerplate structure or near-identical phrasing varied only by a single substituted term, systematically inflates measured retrieval quality, because such passages are unusually easy for both vector search and reranking to match correctly, given how much more similar they are to each other and to a query mentioning the same substituted term than genuinely distinct, independently authored passages would be.

Queries constructed by copying or lightly editing a sentence directly from the intended answer passage similarly inflate measured quality, since the resulting query shares enough exact vocabulary with the one chunk it was derived from that even a naive keyword-matching system, not just a genuine semantic retrieval system, would likely find it, which means such queries fail to meaningfully exercise a system's actual semantic retrieval capability.

Single-document bias occurs when the large majority of a benchmark's gold-relevant evidence is concentrated in one document out of the full corpus, which can make an evaluation appear to validate broad retrieval capability while actually only validating retrieval within that one disproportionately represented document, leaving the system's behavior on the rest of the corpus essentially untested by the benchmark.

Gold label leakage describes a subtler construction pitfall where the structure or phrasing used to write the evaluation question inadvertently reveals which specific chunk is the intended answer, for example by referencing a section heading or an unusual, highly specific phrase that happens to appear nowhere else in the corpus, which again causes the benchmark to measure something closer to exact-phrase matching ability than genuine semantic retrieval.

Easy-question dominance occurs when a benchmark is composed mostly of simple, single-fact, single-chunk queries, which tends to produce a benchmark whose aggregate metrics are high and fairly insensitive to real differences in retrieval configuration, since most of the benchmark's queries are well within reach of almost any reasonable configuration, masking weaknesses that would only show up on harder, multi-part, or terminology-variation queries.

## Collecting Human Relevance Judgments

Human-authored relevance judgments are most reliable when the annotator is given clear, concrete grading criteria rather than an abstract instruction to judge relevance, since different annotators left to interpret "relevant" on their own tend to apply inconsistent standards, some treating any topical overlap as relevant and others reserving a relevant judgment only for chunks that fully and directly answer the query.

Inter-annotator agreement, the degree to which independent annotators assign the same relevance grade to the same query-chunk pair, is a useful diagnostic for whether a benchmark's relevance grading criteria are actually clear and consistently applied; low agreement suggests the grading instructions themselves need to be tightened rather than necessarily indicating that the underlying queries or chunks are unusually ambiguous.

A single annotator constructing an entire benchmark alone, without any form of cross-checking, risks embedding that one annotator's particular interpretation of relevance uniformly across the whole benchmark, which does not necessarily make the benchmark invalid, but means its specific numeric results should be understood as reflecting that one consistent, if idiosyncratic, standard rather than a broadly validated consensus standard.

## Online Evaluation

Offline benchmarks, built from a fixed, pre-labeled set of queries and relevance judgments, have the advantage of being fully reproducible and inexpensive to rerun whenever a system configuration changes, but they are necessarily limited to whatever queries and relevance judgments were included at construction time, which means they cannot by themselves validate how a system performs on the full, evolving distribution of real queries it eventually receives in production.

Online evaluation instead observes real user behavior against a live system, commonly using signals such as click-through rate on returned results, dwell time, or explicit user feedback, to infer which results users found useful without requiring a pre-constructed set of relevance judgments at all, which lets it reflect the actual live query distribution but introduces its own measurement challenges.

Position bias is a central challenge in online evaluation: users are more likely to click on or otherwise engage with a result simply because it appears earlier in the ranked list, independent of that result's actual relevance, which means raw click-through rate at a given rank position cannot be interpreted as a direct relevance signal without first accounting for this positional effect.

A/B testing, where a portion of live traffic is served by a modified retrieval configuration while the remainder continues to be served by the existing configuration, allows an online quality difference to be attributed specifically to the configuration change being tested rather than to any other factor that might also be changing over time, such as a broader shift in the kinds of queries users happen to be submitting during the test period.

## Statistical Significance Considerations

A difference in an aggregate metric observed between two retrieval configurations on a benchmark, or between two arms of an online A/B test, is not automatically a meaningful difference; a benchmark with too few queries, or an online test run for too short a duration or against too little traffic, can easily produce a measured difference that is simply sampling noise rather than a real, reproducible effect of the configuration change being evaluated.

A benchmark with very few queries per category is particularly vulnerable to this problem at the category level even if its total query count looks reasonably large in aggregate, since a conclusion drawn specifically about, for example, table-lookup queries or cross-document queries is only as statistically reliable as however many queries that specific category actually contains, which is a distinct concern from the overall benchmark size.

## Metric Summary

| Metric | What it measures | Blind spot |
| --- | --- | --- |
| Recall@K | Coverage of relevant chunks within the top K | Ignores rank order among the included results |
| Precision@K | Purity of the top K results | Penalizes a wide K even when recall is genuinely high |
| MRR | How early the first relevant result appears | Ignores every relevant result after the first one |
| Hit Rate@K | Whether any relevant result appears at all in top K | Insensitive to how many relevant results exist or their rank |
| nDCG@K | Graded relevance discounted by rank position | Requires reliable, consistently applied graded judgments |

## Constructing Gold Relevance Judgments from a Corpus Directly

A common and reliable way to construct a benchmark's ground truth without hand-assigning chunk identifiers, which would be brittle against any future re-chunking of the corpus, is to anchor each gold relevance judgment to a short, distinctive fragment of the corpus's actual text, together with a stable identifier for which document that fragment belongs to, and resolve that fragment to whichever real, currently persisted chunk happens to contain it at evaluation time.

For this anchoring approach to work correctly, every anchor fragment must appear exactly once within its stated document, since a fragment that matches more than one chunk makes it ambiguous which chunk the gold judgment was actually intended to refer to, and a fragment that does not appear at all, for example because it was paraphrased rather than quoted from the real source text, fails to resolve to any chunk whatsoever and silently drops that piece of ground truth from the benchmark.

This fragment-anchoring approach has a direct practical benefit over hardcoding a chunk's database identifier directly: because chunk identifiers are not stable across re-ingestion, as re-chunking a document can shift chunk boundaries even in unedited sections, a benchmark built around literal chunk identifiers would need to be entirely rebuilt every time its source corpus was re-ingested, whereas a benchmark built around stable document titles and literal text fragments continues to resolve correctly as long as the quoted fragment still exists somewhere in the current version of that document.

## Category-Level Reporting and Bias Auditing

Reporting aggregate metrics across an entire benchmark's full query set can obscure meaningful differences in how a retrieval configuration performs on different kinds of queries, since a configuration that performs very well on simple, direct factual queries but poorly on queries requiring cross-document evidence could still show a respectable overall aggregate score if direct factual queries make up a large share of the benchmark, which is precisely why category labels are assigned to every query and metrics are reported broken down by category rather than only in aggregate.

Auditing a benchmark's category balance before relying on its results means checking that no single category dominates the query count enough to determine the aggregate score almost entirely on its own, that relevant-evidence counts per query vary rather than every query having exactly one gold chunk, and that gold evidence is drawn from a reasonably even spread of documents across the corpus rather than concentrated in just one or two heavily represented documents.

A benchmark that passes this kind of bias audit is more trustworthy specifically because a configuration change's effect on, for example, decomposition-benefiting queries or cross-document queries can be examined in isolation from its effect on simple factual queries, which is a distinction that a single aggregate number across an unbalanced query set would otherwise hide entirely.

## Comparing Configurations Across Multiple Metrics Simultaneously

Comparing two candidate retrieval configurations, for example two different `candidate_limit` and `final_limit` combinations, is rarely a matter of one configuration simply being better than the other on every metric at once; more often one configuration achieves higher recall at the cost of higher latency, while another achieves lower latency at the cost of lower recall, which means the comparison is better framed as examining a trade-off curve across configurations rather than searching for one configuration that dominates every other configuration on every metric simultaneously.

This trade-off framing connects benchmark metrics directly to the candidate-pool-size and final-limit latency and recall figures reported elsewhere in this corpus: a benchmark run across several `candidate_limit` and `final_limit` combinations is, in effect, tracing out exactly this kind of trade-off curve, letting a specific deployment choose the point on that curve that best matches its own latency budget and quality requirements rather than treating any single combination as universally correct.

Reporting a single configuration as "the best" without specifying which metric, or which weighted combination of metrics, that judgment is based on is therefore an incomplete comparison; a defensible configuration recommendation names the specific latency budget or quality floor being optimized against, since a different budget or floor could easily favor a different point along the same underlying trade-off curve.

## Limitations

No single metric in this document fully characterizes retrieval quality on its own; aggregate benchmark reporting in this system intentionally reports several of these metrics together specifically because each one is blind to a different kind of failure, and optimizing a retrieval configuration against only one metric in isolation risks improving that metric while silently regressing another dimension of quality the chosen metric cannot see.

Offline benchmark results, however carefully constructed, remain a proxy for real-world retrieval quality rather than a direct measurement of it, and a benchmark's numeric results should be interpreted as a reproducible signal for comparing configurations against each other under controlled conditions, not as a guaranteed prediction of exactly how the system will perform against the full, unpredictable distribution of real production queries.
