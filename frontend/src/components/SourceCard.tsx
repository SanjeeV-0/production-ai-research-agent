import React, { useState } from 'react';
import { Source } from '../types/research';
import { BookOpen, ChevronDown, ChevronUp, FileText, Hash, Layers } from 'lucide-react';

interface SourceCardProps {
  source: Source;
  index: number;
  isSelected?: boolean;
  onSelect?: () => void;
  additionalContent?: string;
  distance?: number;
  rerankScore?: number;
}

export const SourceCard: React.FC<SourceCardProps> = ({
  source,
  index,
  isSelected,
  onSelect,
  additionalContent,
  distance,
  rerankScore,
}) => {
  const [expanded, setExpanded] = useState(false);

  const toggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    setExpanded(!expanded);
    if (onSelect) onSelect();
  };

  const pagesText = source.page_numbers?.length
    ? source.page_numbers.join(', ')
    : 'N/A';

  return (
    <div
      className={`source-card ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
      style={{
        borderColor: isSelected ? 'var(--primary-indigo)' : undefined,
        boxShadow: isSelected ? '0 0 14px var(--primary-indigo-glow)' : undefined,
      }}
    >
      <div className="source-card-header">
        <div className="source-badge-index">
          {index + 1}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="source-path" title={source.section_path}>
            {source.section_path || 'Document Content'}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.35rem' }}>
            <span className="source-pages">
              <BookOpen size={11} /> Page {pagesText}
            </span>
            {distance !== undefined && (
              <span style={{ fontSize: '0.7rem', color: '#06b6d4', fontFamily: 'var(--font-mono)' }}>
                Dist: {distance.toFixed(4)}
              </span>
            )}
            {rerankScore !== undefined && (
              <span style={{ fontSize: '0.7rem', color: '#10b981', fontFamily: 'var(--font-mono)' }}>
                Score: {rerankScore.toFixed(3)}
              </span>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={toggleExpand}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            padding: '0.2rem',
          }}
          title={expanded ? 'Collapse Metadata' : 'Expand Metadata'}
        >
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Expanded Details Drawer */}
      {expanded && (
        <div className="source-details-drawer">
          <div className="source-detail-item">
            <span className="source-detail-label">
              <Layers size={10} style={{ display: 'inline', marginRight: 2 }} /> Section Path
            </span>
            <span className="source-detail-value" title={source.section_path}>
              {source.section_path || 'N/A'}
            </span>
          </div>

          <div className="source-detail-item">
            <span className="source-detail-label">
              <BookOpen size={10} style={{ display: 'inline', marginRight: 2 }} /> Page(s)
            </span>
            <span className="source-detail-value">
              {pagesText}
            </span>
          </div>

          <div className="source-detail-item">
            <span className="source-detail-label">
              <FileText size={10} style={{ display: 'inline', marginRight: 2 }} /> Document ID
            </span>
            <span className="source-detail-value" title={source.document_id}>
              {source.document_id || 'N/A'}
            </span>
          </div>

          <div className="source-detail-item">
            <span className="source-detail-label">
              <Hash size={10} style={{ display: 'inline', marginRight: 2 }} /> Chunk ID
            </span>
            <span className="source-detail-value" title={source.chunk_id}>
              {source.chunk_id || 'N/A'}
            </span>
          </div>

          {additionalContent && (
            <div style={{ gridColumn: '1 / -1', marginTop: '0.5rem' }}>
              <span className="source-detail-label" style={{ marginBottom: '0.2rem', display: 'block' }}>
                Chunk Snippet
              </span>
              <div
                style={{
                  background: 'rgba(0,0,0,0.4)',
                  padding: '0.6rem 0.8rem',
                  borderRadius: 6,
                  color: '#cbd5e1',
                  fontSize: '0.75rem',
                  lineHeight: '1.5',
                  maxHeight: 150,
                  overflowY: 'auto',
                }}
              >
                {additionalContent}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
