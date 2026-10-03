import React from 'react';
import { DocumentStatus } from '../types/document';
import { CheckCircle2, Clock, AlertTriangle, FileUp, Sparkles } from 'lucide-react';

interface StatusBadgeProps {
  status: DocumentStatus;
  isCurrent?: boolean;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  isCurrent,
  size = 'md',
}) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'READY':
        return {
          icon: <CheckCircle2 size={size === 'sm' ? 12 : 14} />,
          label: 'READY',
          className: 'badge-status-ready',
          description: 'Document indexed & active in vector retrieval',
        };
      case 'PROCESSING':
        return {
          icon: <Clock size={size === 'sm' ? 12 : 14} className="animate-spin" />,
          label: 'PROCESSING',
          className: 'badge-status-processing',
          description: 'Currently extracting sections, chunks, and embeddings',
        };
      case 'FAILED':
        return {
          icon: <AlertTriangle size={size === 'sm' ? 12 : 14} />,
          label: 'FAILED',
          className: 'badge-status-failed',
          description: 'Ingestion failed. Eligible for retry.',
        };
      case 'UPLOADED':
      default:
        return {
          icon: <FileUp size={size === 'sm' ? 12 : 14} />,
          label: 'UPLOADED',
          className: 'badge-status-uploaded',
          description: 'Original file stored, pending extraction pipeline',
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
      <span
        className={`status-pill ${config.className} ${size === 'sm' ? 'status-pill-sm' : ''}`}
        title={config.description}
      >
        {config.icon}
        <span>{config.label}</span>
      </span>

      {isCurrent && (
        <span
          className="current-version-pill"
          title="Current active version queried during retrieval"
        >
          <Sparkles size={11} style={{ marginRight: 3 }} />
          Current
        </span>
      )}
    </div>
  );
};
