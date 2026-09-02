import React, { useState, useEffect } from 'react';
import { Header } from '../components/Header';
import { ResearchInput } from '../components/ResearchInput';
import { AnswerPanel } from '../components/AnswerPanel';
import { SourcesPanel } from '../components/SourcesPanel';
import { TracePanel } from '../components/TracePanel';
import { DownloadButton } from '../components/DownloadButton';
import { LoadingOverlay } from '../components/LoadingOverlay';
import { askResearchQuestion, checkBackendHealth } from '../api/research';
import { ResearchResponse } from '../types/research';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export const ResearchPage: React.FC = () => {
  const [traceEnabled, setTraceEnabled] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [lastQuery, setLastQuery] = useState<string>('');

  useEffect(() => {
    // Check initial health status
    checkBackendHealth().then((healthy) => setIsBackendHealthy(healthy));
  }, []);

  const handleResearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setLastQuery(query);

    try {
      const data = await askResearchQuestion(query, traceEnabled);
      setResponse(data);
      setIsBackendHealthy(true);
    } catch (err: any) {
      console.error('Research request failed:', err);
      setError(err?.message || 'An unexpected error occurred while communicating with the research agent.');
      setIsBackendHealthy(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (lastQuery) {
      handleResearch(lastQuery);
    }
  };

  return (
    <div className="app-container">
      <Header
        traceEnabled={traceEnabled}
        onToggleTrace={setTraceEnabled}
        isBackendHealthy={isBackendHealthy}
      />

      <main style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
        <ResearchInput
          onSearch={handleResearch}
          isLoading={isLoading}
          initialQuery={lastQuery}
        />

        {error && (
          <div className="error-banner">
            <AlertTriangle size={24} style={{ flexShrink: 0, marginTop: 2, color: '#f43f5e' }} />
            <div style={{ flex: 1 }}>
              <h4 className="error-title">Research Execution Error</h4>
              <p className="error-desc">{error}</p>
            </div>
            {lastQuery && (
              <button
                type="button"
                className="btn-secondary"
                onClick={handleRetry}
                style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
              >
                <RefreshCw size={14} /> Retry
              </button>
            )}
          </div>
        )}

        {isLoading ? (
          <LoadingOverlay traceMode={traceEnabled} />
        ) : (
          <>
            <div className="workspace-grid">
              <AnswerPanel
                answer={response?.answer ?? null}
                model={response?.model ?? null}
                isLoading={false}
              />
              <SourcesPanel
                sources={response?.sources ?? []}
                isLoading={false}
                finalResults={response?.trace?.final_results}
              />
            </div>

            {/* Render Download JSON button when a response is present */}
            {response && <DownloadButton response={response} />}

            {/* Render Trace Panel when trace is present in response */}
            {traceEnabled && response?.trace && (
              <TracePanel trace={response.trace} />
            )}
          </>
        )}
      </main>
    </div>
  );
};
