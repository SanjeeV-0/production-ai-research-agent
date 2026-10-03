import {
  ResearchRequest,
  ResearchResponse,
  RetrievalSearchRequest,
  RetrievalSearchResponse,
  BackendHealthResponse,
  BackendReadinessResponse,
} from '../types/research';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';

/**
 * Helper to handle fetch responses and extract meaningful FastAPI error details
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status} (${response.statusText})`;
    try {
      const errorData = await response.json();
      if (errorData?.detail) {
        errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : JSON.stringify(errorData.detail);
      }
    } catch {
      // Fall back to status text
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Sends a research question request to backend POST /research/ask
 */
export async function askResearchQuestion(
  query: string,
  trace: boolean = false
): Promise<ResearchResponse> {
  const payload: ResearchRequest = {
    query: query.trim(),
    trace,
  };

  try {
    const response = await fetch(`${API_BASE}/research/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    return await handleResponse<ResearchResponse>(response);
  } catch (error: any) {
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      throw new Error(
        'Unable to connect to the research agent backend. Please ensure the FastAPI server is running on http://localhost:8000.'
      );
    }
    throw error;
  }
}

/**
 * Directly queries the vector retrieval endpoint POST /retrieval/search
 */
export async function searchRetrievedChunks(
  params: RetrievalSearchRequest
): Promise<RetrievalSearchResponse> {
  const payload: Record<string, unknown> = {
    query: params.query.trim(),
    limit: params.limit ?? 10,
  };

  if (params.document_id) {
    payload.document_id = params.document_id;
  }
  if (params.section_id) {
    payload.section_id = params.section_id;
  }

  try {
    const response = await fetch(`${API_BASE}/retrieval/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    return await handleResponse<RetrievalSearchResponse>(response);
  } catch (error: any) {
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      throw new Error(
        'Unable to connect to the retrieval service. Please verify the backend is running.'
      );
    }
    throw error;
  }
}

/**
 * Checks backend general health: GET /health
 */
export async function checkBackendHealth(): Promise<BackendHealthResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Checks backend database readiness: GET /health/ready
 */
export async function checkBackendReadiness(): Promise<BackendReadinessResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/health/ready`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}
