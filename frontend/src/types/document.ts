/**
 * Document lifecycle types matching the REAL backend response contract
 * (app.ingestion.schemas.DocumentVersionResponse / LogicalDocumentResponse).
 *
 * GET /documents and GET /documents/{logical_document_id} return a
 * LogicalDocumentSummary: a stable logical_document_id plus its current
 * version (nullable -- a document whose only version(s) are still
 * PROCESSING/FAILED has no current version yet). Full version history is a
 * separate call: GET /documents/{logical_document_id}/versions.
 */

export type DocumentStatus = 'UPLOADED' | 'PROCESSING' | 'READY' | 'FAILED';

export interface DocumentVersion {
  id: string; // Document version UUID
  logical_document_id: string; // Grouping UUID for all versions of this document
  version_number: number;
  is_current: boolean;
  status: DocumentStatus;
  title: string;
  document_type: string;
  source?: string | null;
  processing_attempt: number;
  created_at: string;
  updated_at: string;
  processing_started_at?: string | null;
  processing_completed_at?: string | null;
  failed_at?: string | null;
  last_error?: string | null;
}

export interface LogicalDocumentSummary {
  logical_document_id: string;
  current_version: DocumentVersion | null;
}

export interface UploadDocumentMetadata {
  title?: string;
  document_type?: string;
  source?: string;
  // When set, the upload is ingested as a new version of this existing
  // logical document instead of starting a brand new one.
  logical_document_id?: string;
}

// Mirrors app.ingestion.schemas.IngestOutcome. Distinguishes a genuinely new
// document from a detected normalized-content duplicate (global -- matched
// regardless of filename, category, or which logical document it belongs
// to) from a new revision of an explicitly targeted existing document.
export type UploadOutcome = 'created' | 'duplicate' | 'new_version';

// Mirrors app.ingestion.schemas.DocumentUploadResponse -- the response shape
// for POST /documents.
export interface DocumentUploadResult {
  outcome: UploadOutcome;
  version: DocumentVersion;
  updated_metadata_fields: string[];
}
