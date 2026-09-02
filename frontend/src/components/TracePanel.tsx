import React, { useState } from 'react';
import { TraceData, CandidateResult } from '../types/research';
import { Terminal, ChevronDown, ChevronUp, Layers, CheckCircle, AlignLeft, Sparkles, Filter } from 'lucide-react';

interface TracePanelProps {
  trace: TraceData;
}

export const TracePanel: React.FC<TracePanelProps> = ({ trace }) => {
  const [openCandidateId, setOpenCandidateId] = useState<string | null>(null);
  const [openFinalId, setOpenFinalId] = useState<string | null>(null);
  const [showRawContext, setShowRawContext] = useState(false);

  const toggleCandidate = (id: string) => {
    setOpenCandidateId(openCandidateId === id ? null : id);
  };

  const toggleFinal = (id: string) => {
    setOpenFinalId(openFinalId === id ? null : id);
  };

  return (
    <div className="glass-card trace-panel" style={{ padding: '1.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Terminal size={20} color="#6366f1" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.01em' }}>
            Retrieval & Execution Trace
          </h2>
          <span style={{ fontSize: '0.72rem', background: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc', padding: '0.2rem 0.5rem', borderRadius: 4, fontFamily: 'var(--font-mono)' }}>
            Developer Mode
          </span>
        </div>
      </div>

      {/* Summary Metrics Grid */}
      <div className="trace-summary-grid">
        <div className="trace-metric-card">
          <div className="trace-metric-label">Query Requested</div>
          <div style={{ fontSize: '0.9rem', color: '#cbd5e1', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={trace.query}>
            "{trace.query}"
          </div>
        </div>

        <div className="trace-metric-card">
          <div className="trace-metric-label">Candidate Limit</div>
          <div className="trace-metric-value" style={{ color: '#06b6d4' }}>
            {trace.candidate_limit}
          </div>
        </div>

        <div className="trace-metric-card">
          <div className="trace-metric-label">Candidates Retrieved</div>
          <div className="trace-metric-value" style={{ color: '#818cf8' }}>
            {trace.candidates?.length || 0}
          </div>
        </div>

        <div className="trace-metric-card">
          <div className="trace-metric-label">Final Results Selected</div>
          <div className="trace-metric-value" style={{ color: '#10b981' }}>
            {trace.final_results?.length || 0}
          </div>
        </div>
      </div>

      {/* 1. CANDIDATE RETRIEVAL SECTION */}
      <div className="trace-section">
        <div className="trace-section-title">
          <Layers size={14} color="#06b6d4" />
          1. Candidate Retrieval ({trace.candidates?.length || 0} candidates)
        </div>
        <div>
          {trace.candidates && trace.candidates.length > 0 ? (
            trace.candidates.map((cand: CandidateResult, idx: number) => {
              const isOpen = openCandidateId === `cand-${idx}-${cand.chunk_id}`;
              return (
                <div key={`cand-${idx}-${cand.chunk_id}`} className="trace-chunk-accordion">
                  <div
                    className="trace-chunk-header"
                    onClick={() => toggleCandidate(`cand-${idx}-${cand.chunk_id}`)}
                  >
                    <div className="trace-chunk-title">
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#64748b', fontSize: '0.8rem' }}>
                        #{idx + 1}
                      </span>
                      <span>{cand.section_path || 'Section Content'}</span>
                    </div>
                    <div className="trace-scores-row">
                      <span className="score-tag-distance">
                        Distance: {cand.distance?.toFixed(4) ?? 'N/A'}
                      </span>
                      {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>
                  {isOpen && (
                    <div className="trace-chunk-content">
                      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '0.75rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                        <span>Doc ID: <strong style={{ color: '#f8fafc' }}>{cand.document_id}</strong></span>
                        <span>Chunk ID: <strong style={{ color: '#f8fafc' }}>{cand.chunk_id}</strong></span>
                        <span>Pages: <strong style={{ color: '#f8fafc' }}>{cand.page_numbers?.join(', ') || 'N/A'}</strong></span>
                      </div>
                      <div>{cand.content}</div>
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <p style={{ fontSize: '0.85rem', color: '#64748b' }}>No retrieved candidates recorded in trace.</p>
          )}
        </div>
      </div>

      {/* 2. RERANKING & FINAL RESULTS SECTION */}
      <div className="trace-section">
        <div className="trace-section-title">
          <Filter size={14} color="#10b981" />
          2. Cross-Encoder Reranking & Final Selection ({trace.final_results?.length || 0} selected)
        </div>
        <div>
          {trace.final_results && trace.final_results.length > 0 ? (
            trace.final_results.map((res: CandidateResult, idx: number) => {
              const isOpen = openFinalId === `final-${idx}-${res.chunk_id}`;
              return (
                <div key={`final-${idx}-${res.chunk_id}`} className="trace-chunk-accordion" style={{ borderColor: 'rgba(16, 185, 129, 0.3)' }}>
                  <div
                    className="trace-chunk-header"
                    onClick={() => toggleFinal(`final-${idx}-${res.chunk_id}`)}
                  >
                    <div className="trace-chunk-title">
                      <CheckCircle size={14} color="#10b981" />
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#64748b', fontSize: '0.8rem' }}>
                        Selected #{idx + 1}
                      </span>
                      <span>{res.section_path || 'Section Content'}</span>
                    </div>
                    <div className="trace-scores-row">
                      <span className="score-tag-distance">
                        Distance: {res.distance?.toFixed(4) ?? 'N/A'}
                      </span>
                      <span className="score-tag-rerank">
                        Rerank Score: {res.rerank_score?.toFixed(3) ?? 'N/A'}
                      </span>
                      {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>
                  {isOpen && (
                    <div className="trace-chunk-content">
                      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '0.75rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                        <span>Doc ID: <strong style={{ color: '#f8fafc' }}>{res.document_id}</strong></span>
                        <span>Chunk ID: <strong style={{ color: '#f8fafc' }}>{res.chunk_id}</strong></span>
                        <span>Page(s): <strong style={{ color: '#f8fafc' }}>{res.page_numbers?.join(', ') || 'N/A'}</strong></span>
                      </div>
                      <div>{res.content}</div>
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <p style={{ fontSize: '0.85rem', color: '#64748b' }}>No final results selected after reranking.</p>
          )}
        </div>
      </div>

      {/* 3. CONTEXT SENT TO GENERATION */}
      {trace.context && (
        <div className="trace-section">
          <div className="trace-section-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlignLeft size={14} color="#818cf8" />
              3. Context Sent to Generation
            </span>
            <button
              type="button"
              className="btn-secondary"
              style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem' }}
              onClick={() => setShowRawContext(!showRawContext)}
            >
              {showRawContext ? 'Hide Raw Context' : 'View Assembled Text'}
            </button>
          </div>

          {showRawContext ? (
            <div className="context-box">
              {trace.context.text}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {trace.context.sources?.map((src, idx) => (
                <div key={idx} className="trace-chunk-accordion" style={{ padding: '0.85rem 1.2rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem', color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Sparkles size={12} /> [Source {idx + 1}] {src.section_path}
                    </span>
                    <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                      Page {src.page_numbers?.join(', ')}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#cbd5e1', fontFamily: 'var(--font-mono)', lineHeight: 1.5, background: 'rgba(0,0,0,0.3)', padding: '0.6rem 0.8rem', borderRadius: 6 }}>
                    {src.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
