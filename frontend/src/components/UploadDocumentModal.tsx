import React, { useRef, useState } from 'react';
import { UploadCloud, X, AlertTriangle, FileText } from 'lucide-react';
import { uploadDocument, DocumentApiUnavailableError } from '../api/documents';
import { DocumentVersion } from '../types/document';

interface UploadDocumentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploaded: (version: DocumentVersion) => void;
}

const DOCUMENT_TYPES = [
  { value: 'general', label: 'General' },
  { value: 'research_paper', label: 'Research Paper' },
  { value: 'technical_report', label: 'Technical Report' },
  { value: 'manual', label: 'Manual / Documentation' },
];

export const UploadDocumentModal: React.FC<UploadDocumentModalProps> = ({
  isOpen,
  onClose,
  onUploaded,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [documentType, setDocumentType] = useState('general');
  const [source, setSource] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiUnavailable, setApiUnavailable] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const reset = () => {
    setFile(null);
    setTitle('');
    setDocumentType('general');
    setSource('');
    setError(null);
    setApiUnavailable(false);
    setIsUploading(false);
  };

  const handleClose = () => {
    if (isUploading) return;
    reset();
    onClose();
  };

  const handleFileSelected = (selected: File | null) => {
    setFile(selected);
    setError(null);
    setApiUnavailable(false);
    if (selected && !title.trim()) {
      setTitle(selected.name.replace(/\.[^/.]+$/, ''));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || isUploading) return;

    setIsUploading(true);
    setError(null);
    setApiUnavailable(false);

    try {
      const version = await uploadDocument(file, {
        title: title.trim() || file.name,
        document_type: documentType,
        source: source.trim() || undefined,
      });
      onUploaded(version);
      reset();
      onClose();
    } catch (err: any) {
      if (err instanceof DocumentApiUnavailableError) {
        setApiUnavailable(true);
        setError(err.message);
      } else {
        setError(err?.message || 'Upload failed.');
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div
        className="modal-container glass-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-modal-title"
      >
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div className="modal-icon-badge">
              <UploadCloud size={20} />
            </div>
            <h3 id="upload-modal-title" className="modal-title">
              Upload / Ingest Document
            </h3>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={handleClose}
            disabled={isUploading}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <p className="modal-description">
              Select a file to ingest into the knowledge corpus. It will be
              uploaded, chunked, embedded, and made available for retrieval
              once processing completes.
            </p>

            {/* Dropzone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragOver(false);
                const dropped = e.dataTransfer.files?.[0] || null;
                handleFileSelected(dropped);
              }}
              style={{
                marginTop: '1rem',
                padding: '1.5rem',
                borderRadius: 10,
                border: `1.5px dashed ${isDragOver ? '#06b6d4' : 'var(--border-subtle)'}`,
                background: isDragOver ? 'rgba(6,182,212,0.06)' : 'rgba(15,23,42,0.5)',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                hidden
                onChange={(e) => handleFileSelected(e.target.files?.[0] || null)}
              />
              {file ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', color: '#f8fafc' }}>
                  <FileText size={18} color="#06b6d4" />
                  <span style={{ fontWeight: 600 }}>{file.name}</span>
                  <span style={{ color: '#64748b', fontSize: '0.78rem' }}>
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
              ) : (
                <div style={{ color: '#94a3b8' }}>
                  <UploadCloud size={28} style={{ marginBottom: '0.5rem', opacity: 0.7 }} />
                  <div style={{ fontSize: '0.85rem' }}>
                    Drag & drop a file here, or click to browse
                  </div>
                </div>
              )}
            </div>

            {/* Metadata fields */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginTop: '1.25rem' }}>
              <div>
                <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                  Title
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Defaults to filename"
                  disabled={isUploading}
                  style={{
                    width: '100%',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 6,
                    color: '#f8fafc',
                    padding: '0.5rem 0.75rem',
                    fontSize: '0.85rem',
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '0.85rem' }}>
                <div style={{ flex: 1 }}>
                  <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                    Document Type
                  </label>
                  <select
                    value={documentType}
                    onChange={(e) => setDocumentType(e.target.value)}
                    disabled={isUploading}
                    style={{
                      width: '100%',
                      background: 'rgba(15,23,42,0.8)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 6,
                      color: '#f8fafc',
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.85rem',
                    }}
                  >
                    {DOCUMENT_TYPES.map((t) => (
                      <option key={t.value} value={t.value}>
                        {t.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ flex: 1 }}>
                  <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                    Source (optional)
                  </label>
                  <input
                    type="text"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    placeholder="e.g. internal-wiki"
                    disabled={isUploading}
                    style={{
                      width: '100%',
                      background: 'rgba(15,23,42,0.8)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 6,
                      color: '#f8fafc',
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.85rem',
                    }}
                  />
                </div>
              </div>
            </div>

            {error && (
              <div
                className={apiUnavailable ? 'warning-banner' : 'error-banner'}
                style={{ marginTop: '1.25rem', marginBottom: 0 }}
              >
                <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <div className={apiUnavailable ? 'warning-title' : 'error-title'}>
                    {apiUnavailable ? 'Document Management API Not Exposed' : 'Upload Failed'}
                  </div>
                  <div className={apiUnavailable ? 'warning-desc' : 'error-desc'}>
                    {apiUnavailable
                      ? 'The backend does not currently expose an ingestion endpoint (POST /documents). The file was not uploaded.'
                      : error}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-secondary" onClick={handleClose} disabled={isUploading}>
              Cancel
            </button>
            <button type="submit" className="btn-primary" disabled={!file || isUploading}>
              {isUploading ? (
                <>
                  <span className="spinner-ring" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  Uploading...
                </>
              ) : (
                <>
                  <UploadCloud size={15} />
                  Upload & Ingest
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
