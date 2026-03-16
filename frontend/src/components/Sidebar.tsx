import { useCallback } from 'react';
import {
  Upload,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Film,
  ListChecks,
  X,
} from 'lucide-react';
import type { IngestStatus } from '../hooks/useVideoManager';

interface SidebarProps {
  videoPath: string | null;
  fileName:  string | null;
  ingestStatus: IngestStatus;
  statusMessage: string;
  onFileSelect: (file: File) => void;
  onClose?: () => void;
}

const PIPELINE_STEPS = [
  'Upload video file',
  'Extract frames + audio',
  'Transcribe (AWS Transcribe)',
  'Caption scenes (Nova Pro)',
  'Embed frames + captions (Titan)',
  'Build vector indexes',
  'Ready to search & chat',
];

export function Sidebar({
  videoPath,
  fileName,
  ingestStatus,
  statusMessage,
  onFileSelect,
  onClose,
}: SidebarProps) {
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect],
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) onFileSelect(e.target.files[0]);
  };

  const statusIcon = {
    idle:       null,
    uploading:  <Loader2  className="w-3.5 h-3.5 animate-spin text-cyan-400"     />,
    processing: <Loader2  className="w-3.5 h-3.5 animate-spin text-amber-400"   />,
    completed:  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400"          />,
    failed:     <AlertCircle  className="w-3.5 h-3.5 text-red-400"              />,
  }[ingestStatus];

  const statusColor = {
    idle:       'text-gray-500',
    uploading:  'text-cyan-400',
    processing: 'text-amber-400',
    completed:  'text-emerald-400',
    failed:     'text-red-400',
  }[ingestStatus];

  const completedSteps = {
    idle:       0,
    uploading:  1,
    processing: 3,
    completed:  PIPELINE_STEPS.length,
    failed:     2,
  }[ingestStatus];

  return (
    <aside className="w-72 flex flex-col border-r border-gray-800/50 bg-gradient-to-b from-gray-900/80 to-gray-900/40 backdrop-blur-sm shrink-0 overflow-y-auto">

      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-4 border-b border-gray-800/50 shrink-0 bg-gradient-to-r from-transparent to-cyan-900/20">
        <Film className="w-5 h-5 text-cyan-400" />
        <span className="font-bold tracking-tight text-white text-lg">Video RAG</span>
        <span className="ml-auto text-[10px] text-gray-500 bg-cyan-950/50 px-2 py-0.5 rounded-full font-mono border border-cyan-900/50">
          Nova
        </span>
        {onClose && (
          <button
            onClick={onClose}
            title="Close sidebar"
            className="ml-2 p-1.5 rounded-md text-gray-500 hover:text-cyan-300 hover:bg-cyan-900/20 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="flex flex-col gap-6 p-4 flex-1">

        {/* Drop zone */}
        <div
          onDrop={onDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('video-file-input')?.click()}
          className="group flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-cyan-700/40 p-8 cursor-pointer hover:border-cyan-500/70 hover:bg-cyan-900/15 transition-all duration-300 select-none shadow-lg shadow-cyan-900/10"
        >
          <input
            id="video-file-input"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={onInputChange}
          />
          <Upload className="w-8 h-8 text-cyan-500/70 group-hover:text-cyan-400 transition-colors duration-300" />
          <p className="text-sm text-gray-300 text-center leading-snug font-medium">
            Drop a video here<br />
            <span className="text-cyan-400 text-xs font-normal">or click to choose a file</span>
          </p>
          <p className="text-[10px] text-gray-600">MP4 · MOV · MKV · AVI · WEBM</p>
        </div>

        {/* Current video card */}
        {fileName && (
          <div className="rounded-xl bg-gradient-to-br from-gray-800/60 to-gray-800/20 border border-cyan-500/30 p-4 space-y-3 shadow-lg shadow-cyan-900/10">
            <div className="flex items-center gap-2 min-w-0">
              <Film className="w-4 h-4 text-cyan-400 shrink-0" />
              <p className="text-sm font-semibold text-gray-100 truncate">{fileName}</p>
            </div>
            <div className="flex items-center gap-2">
              {statusIcon}
              <span className={`text-sm font-medium ${statusColor}`}>{statusMessage}</span>
            </div>
            {videoPath && (
              <p className="text-[11px] text-gray-500 truncate font-mono bg-gray-900/50 px-2 py-1 rounded border border-gray-700/50" title={videoPath}>
                {videoPath}
              </p>
            )}
          </div>
        )}

        {/* Pipeline steps */}
        <div className="space-y-3">
          <div className="flex items-center gap-1.5 text-xs text-gray-400 font-bold uppercase tracking-widest">
            <ListChecks className="w-4 h-4 text-cyan-500" />
            Processing Pipeline
          </div>
          <div className="space-y-2 bg-gray-900/30 rounded-lg p-3 border border-gray-800/30">
            {PIPELINE_STEPS.map((step, i) => {
              const done    = i < completedSteps;
              const active  = i === completedSteps && ingestStatus === 'processing';
              const failed  = ingestStatus === 'failed' && i === completedSteps;
              return (
                <div key={i} className="flex items-center gap-2.5">
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] shrink-0 font-bold border transition-all duration-300 ${
                      done
                        ? 'bg-emerald-900/60 border-emerald-600 text-emerald-300 shadow-lg shadow-emerald-900/30'
                        : active
                        ? 'bg-cyan-900/60 border-cyan-600 text-cyan-300 shadow-lg shadow-cyan-900/30 animate-pulse'
                        : failed
                        ? 'bg-red-900/60 border-red-600 text-red-300'
                        : 'bg-gray-800 border-gray-600 text-gray-500'
                    }`}
                  >
                    {done ? '✓' : failed ? '✕' : i + 1}
                  </div>
                  <span
                    className={`text-xs font-medium transition-colors duration-300 ${
                      done ? 'text-gray-400' : active ? 'text-cyan-400' : 'text-gray-600'
                    }`}
                  >
                    {step}
                  </span>
                  {active && <Loader2 className="w-3 h-3 animate-spin text-cyan-400 ml-auto shrink-0" />}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </aside>
  );
}
