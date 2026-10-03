/**
 * Document lifecycle types aligned with backend SQLAlchemy models
 * (app.core.models.Document, StoredFile, DocumentStatus)
 */

export type DocumentStatus = 'UPLOADED' | 'PROCESSING' | 'READY' | 'FAILED';

export interface StoredFileMetadata {
  id: string;
  document_id: string;
  original_filename: string;
  content_hash: string;
  size_bytes: number;
  storage_key: string;
}

export interface DocumentVersion {
  id: string; // Document version UUID
  title: string;
  authors?: string | null;
  source?: string | null;
  publication_date?: string | null;
  document_type: string;
  logical_document_id: string; // Grouping UUID for all versions of this document
  content_hash: string;
  version_number: number;
  is_current: boolean;
  status: DocumentStatus;
  processing_started_at?: string | null;
  processing_completed_at?: string | null;
  failed_at?: string | null;
  processing_attempt: number;
  last_error?: string | null;
  document_metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  stored_file?: StoredFileMetadata;
}

export interface LogicalDocumentSummary {
  logical_document_id: string;
  title: string;
  document_type: string;
  source?: string | null;
  current_version_id?: string;
  current_version_number: number;
  current_status: DocumentStatus;
  total_versions: number;
  latest_updated_at: string;
  original_filename?: string;
  file_size_bytes?: number;
  versions: DocumentVersion[];
}

export interface UploadDocumentMetadata {
  title?: string;
  document_type?: string;
  source?: string;
}
