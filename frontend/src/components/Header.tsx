import React from 'react';
import { Cpu, Terminal, Activity } from 'lucide-react';

interface HeaderProps {
  traceEnabled: boolean;
  onToggleTrace: (enabled: boolean) => void;
  isBackendHealthy: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({
  traceEnabled,
  onToggleTrace,
  isBackendHealthy,
}) => {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="brand-icon">
          <Cpu size={22} />
        </div>
        <div>
          <h1 className="brand-title">Production AI Research Agent</h1>
          <p className="brand-subtitle">Retrieval Augmented Generation & Evidence Workspace</p>
        </div>
      </div>

      <div className="header-controls">
        {/* Backend health badge */}
        <div className="badge-status">
          <span className={`status-dot ${isBackendHealthy === false ? 'offline' : ''}`} />
          <Activity size={12} />
          {isBackendHealthy === null
            ? 'Checking API...'
            : isBackendHealthy
            ? 'API Ready'
            : 'API Unreachable'}
        </div>

        {/* Trace toggle switch */}
        <div className="trace-toggle-box">
          <span className="trace-toggle-label">
            <Terminal size={14} />
            Trace Mode
          </span>
          <div
            className={`toggle-switch ${traceEnabled ? 'active' : ''}`}
            onClick={() => onToggleTrace(!traceEnabled)}
            role="button"
            tabIndex={0}
            title={traceEnabled ? 'Trace Mode ON (retrieval details included)' : 'Trace Mode OFF'}
          >
            <div className="toggle-knob" />
          </div>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: traceEnabled ? '#818cf8' : '#64748b' }}>
            {traceEnabled ? 'ON' : 'OFF'}
          </span>
        </div>
      </div>
    </header>
  );
};
