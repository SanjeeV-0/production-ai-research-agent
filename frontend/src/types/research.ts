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
  section_path: string;
  page_numbers: number[];
}

export interface CandidateResult {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number | null;
}

export interface TraceContextSource {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number | null;
}

export interface TraceContext {
  text: string;
  sources: TraceContextSource[];
}

export interface TraceData {
  query: string;
  candidate_limit: number;
  candidates: CandidateResult[];
  final_results: CandidateResult[];
  context: TraceContext | null;
}

export interface ResearchResponse {
  answer: string;
  model: string;
  sources: Source[];
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
}

export interface RetrievedChunkResponse {
  document_id: string;
  chunk_id: string;
  section_id: string;
  section_path: string;
  page_numbers: number[];
  content: string;
  distance: number;
  similarity: number;
  rerank_score: number | null;
}

export interface RetrievalSearchResponse {
  results: RetrievedChunkResponse[];
  trace?: TraceData | null;
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
