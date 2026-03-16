import { useState } from 'react';
import { useChat } from './hooks/useChat';
import { useVideoManager } from './hooks/useVideoManager';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { HealthBadge } from './components/HealthBadge';
import { X, Menu } from 'lucide-react';
import { VideoClip } from './components/VideoClip';

export default function App() {
  const [playerMinimized, setPlayerMinimized] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const {
    messages,
    loading,
    error,
    sendMessage,
    resetMemory,
  } = useChat();

  const {
    videoPath,
    fileName,
    ingestStatus,
    statusMessage,
    handleFile,
  } = useVideoManager();

  const videoReady = ingestStatus === 'completed';

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-950 via-slate-950 to-gray-950 text-gray-100 overflow-hidden font-sans">
      {/* Left sidebar — video management */}
      {sidebarOpen && (
        <Sidebar
          videoPath={videoPath}
          fileName={fileName}
          ingestStatus={ingestStatus}
          statusMessage={statusMessage}
          onFileSelect={handleFile}
          onClose={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="flex flex-col flex-1 min-w-0 overflow-hidden">

        {/* Tab bar */}
        <nav className="flex items-center border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm px-4 shrink-0">
          <button
            onClick={() => setSidebarOpen(v => !v)}
            className="mr-2 p-2 rounded-lg text-gray-400 hover:text-cyan-300 hover:bg-cyan-900/20 transition-colors"
            title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <Menu className="w-4 h-4" />
          </button>

          <h1 className="text-sm font-medium text-cyan-400 px-4 py-3 border-b-2 border-cyan-500">Text Chat</h1>

          {/* Status pill — always visible */}
          {videoPath && (
            <span
              className={`ml-auto text-xs px-3 py-1 rounded-full font-medium transition-all duration-300 ${
                videoReady
                  ? 'bg-emerald-900/50 text-emerald-300 ring-1 ring-emerald-700/50'
                  : ingestStatus === 'failed'
                  ? 'bg-red-900/50 text-red-300 ring-1 ring-red-700/50'
                  : 'bg-amber-900/50 text-amber-300 ring-1 ring-amber-700/50'
              }`}
            >
              {statusMessage}
            </span>
          )}

          {/* API health indicator */}
          <span className={videoPath ? 'ml-3' : 'ml-auto'}>
            <HealthBadge />
          </span>
        </nav>

        {/* Video Player + Chat/Voice Container */}
        <div className="flex flex-1 min-h-0 gap-4 p-4">
          {/* Floating Video Player - Top Left */}
          {videoPath && !playerMinimized && (
            <div className="shrink-0 relative group">
              <div className="w-80 rounded-xl overflow-hidden border border-cyan-500/30 shadow-2xl shadow-cyan-900/30 hover:border-cyan-400/50 transition-all duration-300 bg-black/80 backdrop-blur-sm">
                <div className="relative">
                  <VideoClip clipPath={videoPath} label={fileName || 'Video'} />
                  <button
                    onClick={() => setPlayerMinimized(true)}
                    className="absolute top-2 right-2 p-1.5 bg-gray-900/80 hover:bg-red-900/80 rounded-lg transition-colors duration-200 opacity-0 group-hover:opacity-100"
                    title="Minimize"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <p className="text-[11px] text-gray-500 mt-2 text-center">Video Player</p>
            </div>
          )}

          {/* Minimized Player Button */}
          {videoPath && playerMinimized && (
            <button
              onClick={() => setPlayerMinimized(false)}
              className="shrink-0 w-16 h-20 rounded-lg border border-cyan-500/30 hover:border-cyan-400/50 bg-black/80 backdrop-blur-sm flex items-center justify-center text-xs text-gray-500 hover:text-cyan-400 transition-all duration-200 hover:shadow-lg hover:shadow-cyan-900/30"
              title="Show Player"
            >
              ▶
            </button>
          )}

          {/* Chat Panel - Main Area */}
          <div className="flex-1 min-w-0">
            <ChatPanel
              messages={messages}
              loading={loading}
              error={error}
              videoPath={videoPath}
              videoReady={videoReady}
              onSend={sendMessage}
              onReset={resetMemory}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
