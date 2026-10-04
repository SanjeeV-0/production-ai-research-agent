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
 * The backend implements GET /documents, GET /documents/{id}, and
 * GET /documents/{id}/versions, each returning the shapes declared in
 * `../types/document` (`LogicalDocumentSummary` = logical_document_id +
 * nullable current_version; `DocumentVersion` = one version's lifecycle
 * fields).
 *
 * Every function here still surfaces a 404 as `DocumentApiUnavailableError`
 * instead of silently falling back to mock or cached data, in case a given
 * route genuinely isn't exposed. Callers must handle that case and tell the
 * user the action couldn't be performed rather than pretending it succeeded.
 */

/**
 * Thrown only when a request reaches the server but matches NO route at
 * all (HTTP 404 with FastAPI/Starlette's generic `{"detail": "Not Found"}`
 * body), meaning the endpoint genuinely is not registered by the backend.
 *
 * This is distinct from an application-level 404 raised by a route that
 * DOES exist (e.g. "Document not found: <id>", "Document version not
 * found: <id>") -- those are normal action errors, not evidence that the
 * API is unexposed, and must not be reported as such. It is also distinct
 * from a network failure (server unreachable) and from a real application
 * error (400/409/422/500 from an endpoint that exists).
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
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const errorData = await response.json();
      if (typeof errorData?.detail === 'string') {
        detail = errorData.detail;
      } else if (errorData?.detail) {
        detail = JSON.stringify(errorData.detail);
      }
    } catch {
      // No JSON body to read.
    }

    // FastAPI/Starlette's hardcoded body when no route matches the
    // method+path at all is exactly `{"detail": "Not Found"}`. Every
    // application-level 404 raised by our own route handlers uses a more
    // specific detail message (e.g. "Document not found: <id>"), so this
    // literal match reliably distinguishes "route not exposed" from "this
    // particular resource wasn't found".
    if (response.status === 404 && (detail === undefined || detail === 'Not Found')) {
      throw new DocumentApiUnavailableError();
    }

    throw new Error(detail || `HTTP ${response.status} (${response.statusText})`);
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
 * Fetches a single logical document (identity + current version only).
 * Backend route: GET /documents/{logicalId}
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
 * Fetches the full version history for a logical document, newest first.
 * Backend route: GET /documents/{logicalId}/versions
 */
export async function getDocumentVersions(
  logicalId: string
): Promise<DocumentVersion[]> {
  try {
    const response = await fetch(`${API_BASE}/documents/${logicalId}/versions`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    return await handleResponse<DocumentVersion[]>(response);
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
  if (metadata.logical_document_id) {
    formData.append('logical_document_id', metadata.logical_document_id);
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
 * Backend route: POST /documents/{logicalId}/versions/{versionId}/retry
 */
export async function retryDocumentVersion(
  logicalId: string,
  versionId: string
): Promise<DocumentVersion> {
  try {
    const response = await fetch(
      `${API_BASE}/documents/${logicalId}/versions/${versionId}/retry`,
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
 * Backend route: DELETE /documents/{logicalId}/versions/{versionId}
 */
export async function deleteDocumentVersion(
  logicalId: string,
  versionId: string
): Promise<void> {
  try {
    const response = await fetch(
      `${API_BASE}/documents/${logicalId}/versions/${versionId}`,
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
 * Makes one version the current version of its logical document.
 * Backend route: POST /documents/{logicalId}/versions/{versionId}/set-current
 */
export async function setCurrentVersion(
  logicalId: string,
  versionId: string
): Promise<DocumentVersion> {
  try {
    const response = await fetch(
      `${API_BASE}/documents/${logicalId}/versions/${versionId}/set-current`,
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
