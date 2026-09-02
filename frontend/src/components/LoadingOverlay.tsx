import React, { useEffect, useState } from 'react';
import { Database, Filter, Sparkles, BrainCircuit } from 'lucide-react';

interface LoadingOverlayProps {
  traceMode: boolean;
}

const STAGES = [
  { id: 'retrieve', label: 'Retrieving relevant vector candidates...', icon: Database },
  { id: 'rerank', label: 'Cross-encoder reranking & filtering...', icon: Filter },
  { id: 'assemble', label: 'Assembling document context...', icon: BrainCircuit },
  { id: 'generate', label: 'Generating grounded answer via OpenRouter...', icon: Sparkles },
];

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ traceMode }) => {
  const [currentStageIdx, setCurrentStageIdx] = useState(0);

  useEffect(() => {
    const timer1 = setTimeout(() => setCurrentStageIdx(1), 1200);
    const timer2 = setTimeout(() => setCurrentStageIdx(2), 2400);
    const timer3 = setTimeout(() => setCurrentStageIdx(3), 3600);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  const CurrentIcon = STAGES[currentStageIdx].icon;

  return (
    <div className="loading-overlay glass-card" style={{ marginTop: '1rem' }}>
      <div className="spinner-outer">
        <div className="spinner-ring" />
      </div>

      <div className="loading-text" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <CurrentIcon size={18} color="#818cf8" />
        {STAGES[currentStageIdx].label}
      </div>

      <div className="loading-steps">
        {STAGES.map((stage, idx) => (
          <span
            key={stage.id}
            className={`step-item ${idx <= currentStageIdx ? 'active' : ''}`}
            style={{
              opacity: idx <= currentStageIdx ? 1 : 0.4,
              transition: 'opacity 0.3s ease',
            }}
          >
            {idx > 0 && ' → '}
            {stage.id.toUpperCase()}
          </span>
        ))}
      </div>

      {traceMode && (
        <span style={{ fontSize: '0.72rem', color: '#818cf8', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
          Trace capture enabled • Full retrieval details will be populated below
        </span>
      )}
    </div>
  );
};
