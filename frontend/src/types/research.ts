/**
 * Type definitions matching the backend FastAPI /research/ask contract.
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
  rerank_score: number;
}

export interface TraceContextSource {
  document_id: string;
  chunk_id: string;
  section_id?: string | null;
  section_path: string;
  page_numbers: number[];
  content: string;
  distance: number;
  rerank_score: number;
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
  trace?: TraceData;
}

export interface BackendHealthResponse {
  status: string;
  environment?: string;
}
