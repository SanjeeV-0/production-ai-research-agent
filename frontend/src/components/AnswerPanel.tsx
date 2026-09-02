import React, { useState } from 'react';
import { Sparkles, Copy, Check, Bot } from 'lucide-react';

interface AnswerPanelProps {
  answer: string | null;
  model: string | null;
  isLoading: boolean;
}

export const AnswerPanel: React.FC<AnswerPanelProps> = ({
  answer,
  model,
  isLoading,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!answer) return;
    await navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const wordCount = answer ? answer.trim().split(/\s+/).length : 0;

  return (
    <div className="glass-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Sparkles size={16} color="#818cf8" />
          Answer
        </div>
        {model && (
          <div className="model-tag" title="OpenRouter Generation Model">
            <Bot size={12} />
            {model}
          </div>
        )}
      </div>

      <div className="panel-body" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {isLoading ? (
          <div className="empty-state">
            <div className="spinner-outer" style={{ width: 36, height: 36 }}>
              <div className="spinner-ring" />
            </div>
            <p className="empty-title" style={{ marginTop: '1rem' }}>Generating grounded response...</p>
            <p className="empty-desc">Synthesizing retrieved chunks via LLM pipeline.</p>
          </div>
        ) : answer ? (
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div className="answer-content">
              {answer}
            </div>

            <div className="answer-footer">
              <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                {wordCount} words generated
              </span>
              <button
                type="button"
                className="btn-secondary"
                onClick={handleCopy}
              >
                {copied ? (
                  <>
                    <Check size={14} color="#10b981" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy size={14} />
                    Copy Answer
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <Bot className="empty-icon" />
            <p className="empty-title">No Answer Yet</p>
            <p className="empty-desc">Submit a question above to retrieve context and generate a research answer.</p>
          </div>
        )}
      </div>
    </div>
  );
};
