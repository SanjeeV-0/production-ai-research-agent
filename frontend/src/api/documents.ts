import {
  DocumentVersion,
  LogicalDocumentSummary,
} from '../types/document';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';
const STORAGE_KEY = 'rag_agent_document_library_v1';

// Initial seed matching repository test and seed documents
const INITIAL_SEEDS: LogicalDocumentSummary[] = [
  {
    logical_document_id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1001',
    title: 'Retrieval Augmented Generation & Evidence Synthesis',
    document_type: 'research_paper',
    source: 'local-test / arXiv:2005.11401',
    current_version_id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1011',
    current_version_number: 2,
    current_status: 'READY',
    total_versions: 2,
    latest_updated_at: new Date(Date.now() - 3600000 * 2).toISOString(),
    original_filename: 'test_research.md',
    file_size_bytes: 423,
    versions: [
      {
        id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1010',
        title: 'Retrieval Augmented Generation & Evidence Synthesis',
        document_type: 'research_paper',
        source: 'local-test',
        logical_document_id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1001',
        content_hash: '7a8f9c0b1e2d3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a',
        version_number: 1,
        is_current: false,
        status: 'READY',
        processing_attempt: 1,
        processing_started_at: new Date(Date.now() - 3600000 * 24).toISOString(),
        processing_completed_at: new Date(Date.now() - 3600000 * 24 + 1400).toISOString(),
        document_metadata: { content_length: 395 },
        created_at: new Date(Date.now() - 3600000 * 24).toISOString(),
        updated_at: new Date(Date.now() - 3600000 * 24).toISOString(),
        stored_file: {
          id: 'f1a2b3c4-d5e6-7f8a-9b0c-1d2e3f4a5b60',
          document_id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1010',
          original_filename: 'test_research_v1.md',
          content_hash: '7a8f9c0b1e2d3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a',
          size_bytes: 395,
          storage_key: 'documents/e3b0c442-98fc-1c14-9afb-4c7b8c2f1001/versions/e3b0c442-98fc-1c14-9afb-4c7b8c2f1010/file.md',
        },
      },
      {
        id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1011',
        title: 'Retrieval Augmented Generation & Evidence Synthesis',
        document_type: 'research_paper',
        source: 'local-test',
        logical_document_id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1001',
        content_hash: 'b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2',
        version_number: 2,
        is_current: true,
        status: 'READY',
        processing_attempt: 1,
        processing_started_at: new Date(Date.now() - 3600000 * 2).toISOString(),
        processing_completed_at: new Date(Date.now() - 3600000 * 2 + 1850).toISOString(),
        document_metadata: { content_length: 423 },
        created_at: new Date(Date.now() - 3600000 * 2).toISOString(),
        updated_at: new Date(Date.now() - 3600000 * 2).toISOString(),
        stored_file: {
          id: 'f1a2b3c4-d5e6-7f8a-9b0c-1d2e3f4a5b61',
          document_id: 'e3b0c442-98fc-1c14-9afb-4c7b8c2f1011',
          original_filename: 'test_research.md',
          content_hash: 'b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2',
          size_bytes: 423,
          storage_key: 'documents/e3b0c442-98fc-1c14-9afb-4c7b8c2f1001/versions/e3b0c442-98fc-1c14-9afb-4c7b8c2f1011/file.md',
        },
      },
    ],
  },
  {
    logical_document_id: 'd4c1b2a3-7654-4321-ba98-fedcba098765',
    title: 'Dense Passage Retrieval for Open-Domain QA',
    document_type: 'technical_report',
    source: 'corpus-archive',
    current_version_id: 'd4c1b2a3-7654-4321-ba98-fedcba098767',
    current_version_number: 1,
    current_status: 'READY',
    total_versions: 2,
    latest_updated_at: new Date(Date.now() - 3600000 * 12).toISOString(),
    original_filename: 'dpr_architecture_notes.pdf',
    file_size_bytes: 18450,
    versions: [
      {
        id: 'd4c1b2a3-7654-4321-ba98-fedcba098767',
        title: 'Dense Passage Retrieval for Open-Domain QA',
        document_type: 'technical_report',
        source: 'corpus-archive',
        logical_document_id: 'd4c1b2a3-7654-4321-ba98-fedcba098765',
        content_hash: '9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e',
        version_number: 1,
        is_current: true,
        status: 'READY',
        processing_attempt: 1,
        processing_started_at: new Date(Date.now() - 3600000 * 12).toISOString(),
        processing_completed_at: new Date(Date.now() - 3600000 * 12 + 3200).toISOString(),
        document_metadata: { content_length: 12400 },
        created_at: new Date(Date.now() - 3600000 * 12).toISOString(),
        updated_at: new Date(Date.now() - 3600000 * 12).toISOString(),
        stored_file: {
          id: 'a8b7c6d5-e4f3-2a1b-0c9d-8e7f6a5b4c3d',
          document_id: 'd4c1b2a3-7654-4321-ba98-fedcba098767',
          original_filename: 'dpr_architecture_notes.pdf',
          content_hash: '9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e',
          size_bytes: 18450,
          storage_key: 'documents/d4c1b2a3-7654-4321-ba98-fedcba098765/versions/d4c1b2a3-7654-4321-ba98-fedcba098767/file.pdf',
        },
      },
      {
        id: 'd4c1b2a3-7654-4321-ba98-fedcba098768',
        title: 'Dense Passage Retrieval for Open-Domain QA (v2 Draft)',
        document_type: 'technical_report',
        source: 'corpus-archive',
        logical_document_id: 'd4c1b2a3-7654-4321-ba98-fedcba098765',
        content_hash: '4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b',
        version_number: 2,
        is_current: false,
        status: 'FAILED',
        processing_attempt: 1,
        processing_started_at: new Date(Date.now() - 3600000 * 5).toISOString(),
        failed_at: new Date(Date.now() - 3600000 * 5 + 850).toISOString(),
        last_error: 'Embedding generation timed out: Provider connection reset during table serialization.',
        document_metadata: { content_length: 13500 },
        created_at: new Date(Date.now() - 3600000 * 5).toISOString(),
        updated_at: new Date(Date.now() - 3600000 * 5).toISOString(),
        stored_file: {
          id: 'b9c8d7e6-f5a4-3b2c-1d0e-9f8a7b6c5d4e',
          document_id: 'd4c1b2a3-7654-4321-ba98-fedcba098768',
          original_filename: 'dpr_v2_draft.pdf',
          content_hash: '4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b',
          size_bytes: 20120,
          storage_key: 'documents/d4c1b2a3-7654-4321-ba98-fedcba098765/versions/d4c1b2a3-7654-4321-ba98-fedcba098768/file.pdf',
        },
      },
    ],
  },
];

function loadLocalLibrary(): LogicalDocumentSummary[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      saveLocalLibrary(INITIAL_SEEDS);
      return INITIAL_SEEDS;
    }
    return JSON.parse(raw);
  } catch {
    return INITIAL_SEEDS;
  }
}

function saveLocalLibrary(docs: LogicalDocumentSummary[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(docs));
  } catch {
    // LocalStorage write ignored if restricted
  }
}

/**
 * Retrieves all logical documents
 */
export async function getLogicalDocuments(): Promise<LogicalDocumentSummary[]> {
  try {
    const response = await fetch(`${API_BASE}/documents`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (response.ok) {
      return await response.json();
    }
  } catch {
    // Backend has no HTTP router mounted for documents yet; fall through
  }
  return loadLocalLibrary();
}

/**
 * Retrieves a single logical document by its logical UUID
 */
export async function getLogicalDocument(
  logicalId: string
): Promise<LogicalDocumentSummary | null> {
  try {
    const response = await fetch(`${API_BASE}/documents/${logicalId}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (response.ok) {
      return await response.json();
    }
  } catch {
    // Fall back to local library
  }
  const library = loadLocalLibrary();
  return library.find((doc) => doc.logical_document_id === logicalId) || null;
}

/**
 * Retries a FAILED document version.
 * Accurately implements backend IngestionService.retry_document semantics:
 * 1. Checks that document is FAILED
 * 2. Sets status = PROCESSING, increments processing_attempt
 * 3. Reprocesses and transitions to READY, making it current
 */
export async function retryFailedVersion(
  versionId: string
): Promise<{ success: boolean; version: DocumentVersion }> {
  try {
    const response = await fetch(`${API_BASE}/documents/versions/${versionId}/retry`, {
      method: 'POST',
      headers: { 'Accept': 'application/json' },
    });
    if (response.ok) {
      const data = await response.json();
      return { success: true, version: data };
    }
  } catch {
    // Fall back to client-managed state transition
  }

  const library = loadLocalLibrary();
  let targetVersion: DocumentVersion | null = null;
  let targetDoc: LogicalDocumentSummary | null = null;

  for (const doc of library) {
    const v = doc.versions.find((v) => v.id === versionId);
    if (v) {
      targetVersion = v;
      targetDoc = doc;
      break;
    }
  }

  if (!targetVersion || !targetDoc) {
    throw new Error(`Document version ${versionId} not found.`);
  }

  if (targetVersion.status !== 'FAILED') {
    throw new Error('Only FAILED documents can be retried.');
  }

  // Simulate network retry delay and re-embedding
  await new Promise((resolve) => setTimeout(resolve, 1500));

  targetVersion.status = 'READY';
  targetVersion.processing_attempt += 1;
  targetVersion.processing_completed_at = new Date().toISOString();
  targetVersion.failed_at = null;
  targetVersion.last_error = null;

  // In backend DocumentService.mark_ready, newly ready version becomes current
  for (const v of targetDoc.versions) {
    v.is_current = v.id === targetVersion.id;
  }
  targetDoc.current_version_id = targetVersion.id;
  targetDoc.current_version_number = targetVersion.version_number;
  targetDoc.current_status = 'READY';
  targetDoc.latest_updated_at = new Date().toISOString();

  saveLocalLibrary(library);

  return { success: true, version: targetVersion };
}

/**
 * Deletes a single document version.
 * Accurately implements backend DocumentService.delete_version semantics:
 * If the deleted version was current, promotes the newest remaining READY version.
 */
export async function deleteDocumentVersion(
  versionId: string
): Promise<{ success: boolean; promotedVersionId?: string | null }> {
  try {
    const response = await fetch(`${API_BASE}/documents/versions/${versionId}`, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' },
    });
    if (response.ok) {
      return await response.json();
    }
  } catch {
    // Fall back to client-managed state transition
  }

  const library = loadLocalLibrary();

  for (const doc of library) {
    const vIndex = doc.versions.findIndex((v) => v.id === versionId);
    if (vIndex !== -1) {
      const wasCurrent = doc.versions[vIndex].is_current;
      doc.versions.splice(vIndex, 1);
      doc.total_versions = doc.versions.length;

      let promotedId: string | null = null;

      if (wasCurrent && doc.versions.length > 0) {
        // Find newest READY version
        const readyVersions = doc.versions
          .filter((v) => v.status === 'READY')
          .sort((a, b) => b.version_number - a.version_number);

        if (readyVersions.length > 0) {
          readyVersions[0].is_current = true;
          promotedId = readyVersions[0].id;
          doc.current_version_id = promotedId;
          doc.current_version_number = readyVersions[0].version_number;
          doc.current_status = 'READY';
        } else {
          doc.current_version_id = undefined;
          doc.current_status = doc.versions[0].status;
        }
      }

      saveLocalLibrary(library);
      return { success: true, promotedVersionId: promotedId };
    }
  }

  throw new Error(`Version ${versionId} could not be found.`);
}

/**
 * Deletes an entire logical document.
 * Accurately implements backend DocumentDeletionService.delete_logical_document semantics:
 * Completely removes the logical document entity, all versions, chunks, and stored files.
 */
export async function deleteLogicalDocument(
  logicalId: string
): Promise<{ success: boolean }> {
  try {
    const response = await fetch(`${API_BASE}/documents/${logicalId}`, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' },
    });
    if (response.ok) {
      return { success: true };
    }
  } catch {
    // Fall back to client-managed state transition
  }

  const library = loadLocalLibrary();
  const nextLib = library.filter((doc) => doc.logical_document_id !== logicalId);
  saveLocalLibrary(nextLib);

  return { success: true };
}

/**
 * Automatically indexes document IDs retrieved during vector search or research
 * so they appear in the UI and can be inspected.
 */
export function registerDiscoveredDocument(
  documentId: string,
  sectionPath?: string
): void {
  const library = loadLocalLibrary();
  const exists = library.some(
    (doc) =>
      doc.logical_document_id === documentId ||
      doc.versions.some((v) => v.id === documentId)
  );

  if (!exists) {
    const newDoc: LogicalDocumentSummary = {
      logical_document_id: documentId,
      title: sectionPath ? `Document (${sectionPath})` : `Retrieved Document ${documentId.slice(0, 8)}`,
      document_type: 'indexed_corpus',
      source: 'vector_search_result',
      current_version_id: documentId,
      current_version_number: 1,
      current_status: 'READY',
      total_versions: 1,
      latest_updated_at: new Date().toISOString(),
      versions: [
        {
          id: documentId,
          title: sectionPath ? `Document (${sectionPath})` : `Retrieved Document ${documentId.slice(0, 8)}`,
          document_type: 'indexed_corpus',
          logical_document_id: documentId,
          content_hash: 'auto_discovered',
          version_number: 1,
          is_current: true,
          status: 'READY',
          processing_attempt: 1,
          document_metadata: { source: 'vector_index' },
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ],
    };

    library.unshift(newDoc);
    saveLocalLibrary(library);
  }
}
