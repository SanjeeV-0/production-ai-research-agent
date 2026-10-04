/**
 * Type definitions strictly matching backend FastAPI contracts:
 * - app/api/research.py (POST /research/ask)
 * - app/api/retrieval.py (POST /retrieval/search)
 * - app/main.py (GET /health, GET /health/ready)
 */

export interface ResearchRequest {
  query: string;
  trace?: boolean;
}

export interface Source {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string | null;
  page_numbers: number[];
}

export interface CandidateResult {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string | null;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number | null;
}

export interface TraceContextSource {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string | null;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number | null;
}

export interface TraceContext {
  text: string;
  sources: TraceContextSource[];
}

/**
 * Debug trace for one retrieval operation. The retrieval pipeline always
 * decomposes `original_query` into 1-5 `sub_queries` (deduplicated, the
 * original query always included; decomposition only occurs for multiple
 * distinct information needs -- a single-intent query simply yields
 * `sub_queries == [original_query]`). Retrieval runs independently per
 * sub-query, candidates are merged/deduplicated across all of them
 * (`raw_candidate_count` before merge, `deduplicated_candidate_count`
 * after), and exactly one global reranker pass picks the final Top K from
 * the merged pool. The frontend only displays this -- it never re-derives
 * or re-runs any of this logic itself.
 */
export interface TraceData {
  query: string;
  original_query: string;
  sub_queries: string[];
  candidate_limit: number;
  raw_candidate_count: number;
  deduplicated_candidate_count: number;
  candidates: CandidateResult[];
  final_results: CandidateResult[];
  context: TraceContext | null;
}

/**
 * Why trace data is or isn't present on a response. `TRACE_ENABLED` is a
 * server-side capability ceiling; request-level `trace` is the per-request
 * ask. The server only ever captures a trace when both are true
 * (`request.trace AND settings.trace_enabled`) -- neither ever controls
 * whether query decomposition or retrieval itself runs, only whether that
 * execution's trace is recorded and returned.
 */
export type TraceUnavailableReason = 'server_disabled';

export interface ResearchResponse {
  answer: string;
  model: string;
  sources: Source[];
  trace_requested: boolean;
  trace_available: boolean;
  trace_unavailable_reason: TraceUnavailableReason | null;
  trace?: TraceData | null;
}

/**
 * Contracts for vector retrieval endpoint (POST /retrieval/search)
 */
export interface RetrievalSearchRequest {
  query: string;
  limit?: number; // 1 to 50, default 10
  document_id?: string | null;
  section_id?: string | null;
  // Per-request ask for trace capture. Combined with the server's
  // TRACE_ENABLED setting as request.trace AND settings.trace_enabled --
  // see `RetrievalSearchResponse` for how the server reports the outcome.
  trace?: boolean;
}

export interface RetrievedChunkResponse {
  document_id: string;
  chunk_id: string;
  section_id: string | null;
  section_path: string | null;
  page_numbers: number[];
  content: string;
  distance: number;
  similarity: number;
  rerank_score: number | null;
}

export interface RetrievalSearchResponse {
  results: RetrievedChunkResponse[];
  trace_requested: boolean;
  trace_available: boolean;
  trace_unavailable_reason: TraceUnavailableReason | null;
  trace: TraceData | null;
}

/**
 * System health contracts
 */
export interface BackendHealthResponse {
  status: string;
  environment?: string;
}

export interface BackendReadinessResponse {
  status: 'ready' | 'not_ready';
  checks: {
    database: boolean;
  };
}
