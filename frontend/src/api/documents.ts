import {
  DocumentVersion,
  LogicalDocumentSummary,
  UploadDocumentMetadata,
} from '../types/document';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';

/**
 * This module is the single boundary between the UI and the backend's
 * document-management HTTP API.
 *
 * IMPORTANT: As of now, the FastAPI backend does not mount any HTTP routes
 * for document management (`app/main.py` only registers `/research` and
 * `/retrieval`). The endpoints called below (`/documents`, ...) describe the
 * INTENDED contract for that future API — they are not guaranteed to exist.
 *
 * Every function here surfaces that gap explicitly via
 * `DocumentApiUnavailableError` instead of silently falling back to mock or
 * cached data. Callers must handle that case and tell the user the backend
 * API isn't exposed yet rather than pretending an action succeeded.
 */

/**
 * Thrown when a document-management request reaches the server but matches
 * no route (HTTP 404), meaning the endpoint is not currently exposed by the
 * backend. Distinct from a network failure (server unreachable) and from a
 * real application error (e.g. 400/422/500 from an endpoint that exists).
 */
export class DocumentApiUnavailableError extends Error {
  constructor(
    message = 'Document management API is not currently exposed by the backend.'
  ) {
    super(message);
    this.name = 'DocumentApiUnavailableError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (response.status === 404) {
    throw new DocumentApiUnavailableError();
  }

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status} (${response.statusText})`;
    try {
      const errorData = await response.json();
      if (errorData?.detail) {
        errorMessage =
          typeof errorData.detail === 'string'
            ? errorData.detail
            : JSON.stringify(errorData.detail);
      }
    } catch {
      // Fall back to status text
    }
    throw new Error(errorMessage);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

function wrapError(error: unknown): never {
  if (error instanceof DocumentApiUnavailableError) {
    throw error;
  }
  if (
    error instanceof TypeError &&
    typeof error.message === 'string' &&
    error.message.includes('Failed to fetch')
  ) {
    throw new Error(
      'Unable to reach the backend server. Please ensure the FastAPI server is running.'
    );
  }
  throw error;
}

/**
 * Lists logical documents. Intended backend route: GET /documents
 */
export async function listDocuments(): Promise<LogicalDocumentSummary[]> {
  try {
    const response = await fetch(`${API_BASE}/documents`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    return await handleResponse<LogicalDocumentSummary[]>(response);
  } catch (error) {
    return wrapError(error);
  }
}

/**
 * Fetches a single logical document and its versions.
 * Intended backend route: GET /documents/{logicalId}
 */
export async function getDocument(
  logicalId: string
): Promise<LogicalDocumentSummary> {
  try {
    const response = await fetch(`${API_BASE}/documents/${logicalId}`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    return await handleResponse<LogicalDocumentSummary>(response);
  } catch (error) {
    return wrapError(error);
  }
}

/**
 * Uploads a file for ingestion via the backend's IngestionService.
 * Intended backend route: POST /documents (multipart/form-data)
 */
export async function uploadDocument(
  file: File,
  metadata: UploadDocumentMetadata = {}
): Promise<DocumentVersion> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('title', metadata.title?.trim() || file.name);
  formData.append('document_type', metadata.document_type || 'general');
  if (metadata.source?.trim()) {
    formData.append('source', metadata.source.trim());
  }

  try {
    const response = await fetch(`${API_BASE}/documents`, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      body: formData,
    });
    return await handleResponse<DocumentVersion>(response);
  } catch (error) {
    return wrapError(error);
  }
}

/**
 * Retries a FAILED document version.
 * Intended backend route: POST /documents/versions/{versionId}/retry
 */
export async function retryDocumentVersion(
  versionId: string
): Promise<DocumentVersion> {
  try {
    const response = await fetch(
      `${API_BASE}/documents/versions/${versionId}/retry`,
      {
        method: 'POST',
        headers: { Accept: 'application/json' },
      }
    );
    return await handleResponse<DocumentVersion>(response);
  } catch (error) {
    return wrapError(error);
  }
}

/**
 * Deletes a single document version.
 * Intended backend route: DELETE /documents/versions/{versionId}
 */
export async function deleteDocumentVersion(versionId: string): Promise<void> {
  try {
    const response = await fetch(
      `${API_BASE}/documents/versions/${versionId}`,
      {
        method: 'DELETE',
        headers: { Accept: 'application/json' },
      }
    );
    await handleResponse<void>(response);
  } catch (error) {
    wrapError(error);
  }
}

/**
 * Deletes an entire logical document (all versions, chunks, and files).
 * Intended backend route: DELETE /documents/{logicalId}
 */
export async function deleteLogicalDocument(logicalId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/documents/${logicalId}`, {
      method: 'DELETE',
      headers: { Accept: 'application/json' },
    });
    await handleResponse<void>(response);
  } catch (error) {
    wrapError(error);
  }
}
