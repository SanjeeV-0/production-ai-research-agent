import React from 'react';
import { Source, CandidateResult } from '../types/research';
import { SourceCard } from './SourceCard';
import { Database, AlertCircle } from 'lucide-react';

interface SourcesPanelProps {
  sources: Source[];
  isLoading: boolean;
  finalResults?: CandidateResult[];
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({
  sources,
  isLoading,
  finalResults,
}) => {
  // Map final result details if trace is available
  const getTraceInfo = (chunkId: string) => {
    if (!finalResults) return undefined;
    return finalResults.find((res) => res.chunk_id === chunkId);
  };

  return (
    <div className="glass-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Database size={16} color="#06b6d4" />
          Sources & Evidence
        </div>
        {sources.length > 0 && (
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#06b6d4', background: 'rgba(6,182,212,0.1)', padding: '0.2rem 0.6rem', borderRadius: 9999 }}>
            {sources.length} {sources.length === 1 ? 'Source' : 'Sources'}
          </span>
        )}
      </div>

      <div className="panel-body" style={{ flex: 1 }}>
        {isLoading ? (
          <div className="empty-state">
            <div className="spinner-outer" style={{ width: 36, height: 36 }}>
              <div className="spinner-ring" style={{ borderTopColor: '#06b6d4' }} />
            </div>
            <p className="empty-title" style={{ marginTop: '1rem' }}>Retrieving Supporting Context...</p>
            <p className="empty-desc">Searching vector database & reranking candidates.</p>
          </div>
        ) : sources.length > 0 ? (
          <div className="sources-list">
            {sources.map((source, index) => {
              const traceInfo = getTraceInfo(source.chunk_id);
              return (
                <SourceCard
                  key={source.chunk_id || index}
                  source={source}
                  index={index}
                  additionalContent={traceInfo?.content}
                  distance={traceInfo?.distance}
                  rerankScore={traceInfo?.rerank_score}
                />
              );
            })}
          </div>
        ) : (
          <div className="empty-state">
            <AlertCircle className="empty-icon" />
            <p className="empty-title">No Supporting Sources</p>
            <p className="empty-desc">
              When a query is submitted, retrieved document chunks and section pathways will be displayed here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
