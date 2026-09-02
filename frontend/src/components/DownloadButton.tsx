import React from 'react';
import { Download, Check } from 'lucide-react';
import { ResearchResponse } from '../types/research';

interface DownloadButtonProps {
  response: ResearchResponse | null;
}

export const DownloadButton: React.FC<DownloadButtonProps> = ({ response }) => {
  const [downloaded, setDownloaded] = React.useState(false);

  if (!response) return null;

  const handleDownload = () => {
    // Generate ISO timestamp string: research_result_2026-09-02.json
    const dateStr = new Date().toISOString().split('T')[0];
    const filename = `research_result_${dateStr}.json`;

    // Pretty-print exact API response
    const jsonString = JSON.stringify(response, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2500);
  };

  return (
    <div className="action-bar">
      <button
        type="button"
        className="btn-secondary"
        onClick={handleDownload}
        style={{
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(6, 182, 212, 0.15))',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          color: '#e0e7ff',
          padding: '0.65rem 1.25rem',
          fontSize: '0.88rem',
          fontWeight: 600,
        }}
      >
        {downloaded ? (
          <>
            <Check size={16} color="#10b981" />
            Downloaded JSON!
          </>
        ) : (
          <>
            <Download size={16} color="#818cf8" />
            Download JSON
          </>
        )}
      </button>
    </div>
  );
};
