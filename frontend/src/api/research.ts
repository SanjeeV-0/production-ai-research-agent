import { ResearchRequest, ResearchResponse, BackendHealthResponse } from '../types/research';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';

/**
 * Sends a research question request to the backend /research/ask endpoint.
 */
export async function askResearchQuestion(
  query: string,
  trace: boolean = false
): Promise<ResearchResponse> {
  const payload: ResearchRequest = {
    query,
    trace,
  };

  const endpoint = `${API_BASE}/research/ask`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorMessage = `Server error (${response.status} ${response.statusText})`;
      try {
        const errorData = await response.json();
        if (errorData?.detail) {
          errorMessage = typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail);
        }
      } catch {
        // Fallback to HTTP status message
      }
      throw new Error(errorMessage);
    }

    const data: ResearchResponse = await response.json();
    return data;
  } catch (error: any) {
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      throw new Error('Unable to connect to research agent backend. Please ensure FastAPI server is running at http://localhost:8000.');
    }
    throw error;
  }
}

/**
 * Checks backend health readiness.
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
    });
    if (!response.ok) return false;
    const data: BackendHealthResponse = await response.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}
