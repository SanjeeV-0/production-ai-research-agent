# Query Decomposition in Practice

## Introduction

A single dense vector can represent a single coherent idea reasonably well, but a user question that actually contains several distinct sub-questions, such as a comparison between two approaches across several criteria, does not compress cleanly into one vector without blurring together the separate pieces of evidence needed to answer each part.

Query decomposition exists to address this mismatch: rather than forcing one embedding to stand in for a multi-part question, the question is split into several narrower subqueries, each of which is embedded and searched independently, so that evidence relevant to one part of the original question is not forced to compete for representation inside a single averaged vector alongside evidence for an entirely different part.

This document describes how decomposition is generated in this system, how its results are merged back together, when decomposition genuinely helps retrieval, when it does not, and the quality pitfalls that show up once decomposition is driven by a language model operating on live, unpredictable user queries rather than on a curated set of examples.

## How Decomposition Is Generated

Decomposition in this system is performed by an OpenRouter-backed large language model, which is given the original user query and asked to produce a small set of retrieval-oriented subqueries that, taken together, cover the distinct pieces of information the original query is asking for, without being instructed to answer the question itself.

The original query is always retained alongside whatever subqueries are generated, never replaced by them, because the original query remains the text that the cross-encoder reranker will ultimately score every candidate against, so even when decomposition substantially rewrites or splits the query for the purpose of vector search, the reranking stage still evaluates relevance with respect to what the user actually asked.

Generated subqueries are retrieval-oriented rather than conversational: a well-formed subquery is phrased as something that could plausibly match a chunk of source documentation directly, rather than as a sub-question still addressed to a conversational assistant, since the subquery's only job in this architecture is to be embedded and searched, not to be read or answered by anything downstream.

## Independent Retrieval and Merging

Every subquery produced by decomposition, along with the retained original query, is embedded and searched against the vector index completely independently, meaning the vector search stage has no awareness that several of its queries in a given request are related subquestions of a single original question rather than unrelated queries submitted separately.

After independent retrieval, every candidate chunk returned by any subquery is combined into one pool and deduplicated by chunk identifier, with the lowest vector distance observed for a given chunk across all subqueries retained as that chunk's representative distance going into the merged pool, which is the same merge rule described from the architectural perspective elsewhere in this corpus.

This merge behavior means a chunk can earn a strong position in the candidate pool by matching well against just one subquery, even if it would not have matched well against the original query taken as a whole, which is precisely the mechanism that lets decomposition recover evidence for one specific part of a multi-part question that the original, undecomposed query's embedding would have diluted.

## Reranking Against the Original Query

A distinctive design choice in this architecture is that the cross-encoder reranker always scores the merged candidate pool against the original user query, never against any of the generated subqueries, even for candidates that were only surfaced because they matched a specific subquery well during vector search.

This choice reflects the reranker's role in the pipeline: vector search's job is to maximize the chance that relevant evidence makes it into the candidate pool at all, where decomposition helps by giving each distinct piece of the question its own dedicated retrieval pass, while reranking's job is to judge the actual relevance of each candidate to what the user genuinely asked, which is better served by the full, original question than by a narrower subquery fragment that may have lost context present in the original phrasing.

A practical consequence of this choice is that a candidate chunk can be ranked relatively low by the final reranking stage even though it matched a subquery very well during vector search, if that chunk turns out to be less relevant to the original query's full intent than its subquery-level match alone suggested, which is intentional: subquery matching is a recall mechanism for assembling candidates, not a substitute for judging final relevance against what was actually asked.

## When Decomposition Helps

Comparative questions asking about differences between two or more named approaches across several criteria are a clear case where decomposition helps, because each criterion, such as accuracy, latency, or operational complexity, may be discussed in a different part of a document or even in a different document entirely, and a single embedding of the full comparative question would have to average across all of these criteria at once rather than retrieving targeted evidence for each one.

Conjunctive multi-part questions, phrased with an explicit "and" joining two genuinely separate asks, such as a question asking both what causes a given failure mode and what mitigation is described for it, similarly benefit from decomposition, since the cause and the mitigation may be described in physically separate paragraphs or sections that a single combined query embedding would not retrieve with equal strength.

Multi-hop questions, where answering the question fully requires combining a fact established in one part of the corpus with a separate fact established elsewhere, benefit from decomposition specifically because each hop can be issued as its own targeted subquery, increasing the chance that evidence for both hops lands in the candidate pool rather than only evidence for whichever hop happens to dominate a single combined embedding.

## When Decomposition Does Not Help

Simple, direct factual questions asking for a single fact or definition generally do not benefit from decomposition, since there is only one coherent idea for a single embedding to represent in the first place, and asking a language model to decompose such a query typically either returns the original query unchanged as its only subquery or produces a small number of near-duplicate paraphrases that search for essentially the same thing the original query already would have found on its own.

Decomposition's latency cost, dominated by the network round trip to the decomposition language model rather than by the additional retrieval work it causes, is paid on every query that is routed through decomposition regardless of whether that query actually benefits from being split, which means invoking decomposition for a query that did not need it is pure latency overhead with no corresponding quality gain.

A complex-looking question is not automatically a question that benefits from decomposition; a single, precisely phrased question that already names the specific concept it is asking about, even if that question is long or technically dense, can be fully answerable by a single well-targeted vector search, and in such cases decomposing it into several narrower subqueries risks fragmenting a search that was already well aimed into several less-targeted ones, each matching a smaller part of the original concept.

## Decomposition Quality Pitfalls

Over-decomposition occurs when a language model splits a question into more subqueries than the question's actual distinct information needs warrant, which increases the raw number of vector searches performed and the raw size of the pre-deduplication candidate pool without a corresponding increase in retrieval quality, since several near-identical subqueries tend to surface largely overlapping candidates rather than new evidence.

Subquery drift occurs when a generated subquery gradually loses connection to the specific intent of the original query, phrasing something that is topically related but no longer actually addresses what the user asked, which can introduce candidates into the pool that are thematically plausible but not genuinely relevant, candidates the reranker must then correctly demote using the original query rather than being misled by their strong subquery-level vector match.

Hallucinated subqueries are a more severe version of drift, where the decomposition model introduces an assumption, a named entity, or a sub-topic that was not actually present in or implied by the original query at all, which can happen when a language model pattern-matches a query to a familiar template of comparative or multi-part questions even when the actual query does not fit that template, producing subqueries that search for something the user never asked about.

OpenRouter call failures and timeouts are an operational pitfall rather than a quality pitfall: a decomposition request that times out or returns an error must fall back to treating the original query as the only subquery, rather than failing the entire retrieval request, since decomposition is explicitly an optional enhancement stage layered in front of a pipeline that is fully capable of operating on the original query alone.

## Matching Decomposition to Query Type

| Query type | Decomposition generally helps | Primary reason |
| --- | --- | --- |
| Single-fact direct question | No | One coherent idea, nothing to split |
| Definition or terminology question | No | Narrow, already well-targeted embedding |
| Comparative, multi-criteria question | Yes | Each criterion may live in a different passage |
| Conjunctive "what is X and what is Y" question | Yes | X and Y may be described in separate sections |
| Multi-hop question requiring two combined facts | Yes | Each hop benefits from its own targeted search |
| Long but single-concept technical question | Not necessarily | Already well aimed at one specific concept |

This table reflects a general tendency rather than a deterministic rule, since the decomposition language model makes its own judgment call on each incoming query at request time, and a query that superficially resembles one row of this table can still be decomposed differently in practice depending on its exact phrasing.

## Interaction With Candidate Pool Size

Decomposition directly increases the raw volume of vector search results produced before deduplication, since every additional subquery contributes its own independently ranked list of up to `candidate_limit` results, and only after all of these lists are merged and deduplicated by chunk identifier does the pool shrink back down to its effective size for reranking.

Because the reranker still only ever scores the deduplicated pool, not the raw pre-merge volume, decomposition's effect on reranking latency is bounded by how much genuine overlap exists between subqueries' results rather than by the number of subqueries directly; subqueries that are highly redundant with each other produce a merged pool not much larger than a single query's results would have, while subqueries that are genuinely distinct from each other produce a merged pool that can approach, or in principle reach, the sum of each subquery's individual result count.

This means over-decomposition's main operational cost is not necessarily a larger reranking workload, since redundant subqueries tend to collapse back down during deduplication, but the wasted latency and resource cost of issuing and embedding queries that largely rediscover candidates another subquery would have found anyway.

## Prompting Design for the Decomposition Model

The prompt given to the decomposition language model includes explicit instructions to produce retrieval-oriented subqueries rather than conversational sub-questions, and to avoid introducing any named entity, assumption, or sub-topic that was not already present in or directly implied by the original query, which is the primary mitigation applied against the hallucinated-subquery pitfall described elsewhere in this document.

Few-shot examples included in the prompt demonstrate the intended decomposition behavior for both decomposable and non-decomposable queries side by side, specifically so the model has a concrete pattern to follow for recognizing when a query should be returned essentially unchanged as a single subquery rather than being split, rather than relying on the model's own undirected judgment of the general concept of decomposition alone.

The decomposition model is run at a low sampling temperature rather than a higher, more creative setting, since a lower temperature favors the most probable, conservative completion the model would produce for a given query, which is a deliberate choice to reduce the chance of the model introducing novel or unexpected subquery phrasing that drifts from the original query's literal intent, at the acknowledged cost of somewhat less varied subquery phrasing than a higher-temperature setting might produce.

## Decomposition Count Limits and Cost Control

An upper limit is enforced on the number of subqueries a single decomposition call is allowed to produce, capping decomposition at four subqueries in addition to the always-retained original query, which bounds both the number of independent vector searches a single request can trigger and the resulting pre-deduplication candidate volume that the merge step has to process.

This cap exists primarily as a cost and latency control rather than as a reflection of some natural limit on how many genuinely distinct sub-questions a complex query could be broken into, since a small number of real-world queries could plausibly be decomposed into more than four meaningfully distinct subqueries, and such queries are simply decomposed down to whatever the model judges to be the four most useful subqueries within the enforced limit rather than being allowed to expand further.

Because each additional subquery contributes its own independent vector search and its own contribution to the pre-deduplication candidate pool, the subquery count cap is the most direct lever available for bounding the worst-case additional load a single decomposed request can place on the vector search stage, independent of whatever `candidate_limit` is configured for each individual subquery's own search.

## Decomposition in Multi-Turn Contexts

Decomposition in this system operates on the current turn's query text alone; it does not fold in prior conversation history, so a short follow-up query that only makes sense in light of an earlier turn, such as a brief clarifying question referring back to something discussed previously without restating it, is decomposed purely based on its own literal text, with no access to whatever earlier context would actually be needed to interpret it correctly.

This means a follow-up query that depends on unstated context from a previous turn can be decomposed into subqueries that are reasonable interpretations of the follow-up's literal wording but miss the actual intent the user had in mind, since the decomposition model has no mechanism in this architecture to reach back into conversation history to recover the missing context before deciding how to split the query.

Addressing this limitation, if a deployment required it, would require resolving a follow-up query against prior conversation turns into a single, self-contained query before that resolved query ever reaches the decomposition stage described in this document, which is a distinct concern from decomposition itself and is not something the decomposition stage as currently designed attempts to handle on its own.

## Limitations

Decomposition quality is entirely dependent on the capability and consistency of the underlying language model invoked for that stage; there is no deterministic or rule-based decomposition fallback in this system, so a model that is temporarily degraded in quality, without actually failing or timing out, can produce subqueries that are less useful than the original query alone, without that degradation being distinguishable from an ordinary, necessary decomposition at the architectural level.

There is no mechanism that evaluates, after the fact, whether a given decomposition actually improved the final reranked result compared to what the original query alone would have retrieved; decomposition is applied prospectively based on the language model's judgment at request time, not validated retrospectively against the eventual retrieval outcome for that specific request.
