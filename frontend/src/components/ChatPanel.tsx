import { useRef, useEffect, useState, useCallback } from 'react';
import { Send, Trash2, Loader2, Sparkles, ImagePlus, X } from 'lucide-react';
import { MessageBubble } from './MessageBubble';
import type { ChatMessage } from '../api/types';

interface ChatPanelProps {
  messages:   ChatMessage[];
  loading:    boolean;
  error:      string | null;
  videoPath:  string | null;
  videoReady: boolean;
  onSend:     (text: string, videoPath?: string, imageBase64?: string) => void;
  onReset:    () => void;
}

const SUGGESTIONS = [
  'Find the most exciting moment',
  'What happens at the beginning?',
  'Show me any action scenes',
  'Summarise the video content',
  'Find a clip with dialogue',
];

export function ChatPanel({
  messages,
  loading,
  error,
  videoPath,
  videoReady,
  onSend,
  onReset,
}: ChatPanelProps) {
  const [input, setInput]               = useState('');
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64]   = useState<string | null>(null);
  const bottomRef   = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll on new messages.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if ((!text && !imageBase64) || loading) return;
    onSend(text || 'Find a clip similar to this image', videoPath ?? undefined, imageBase64 ?? undefined);
    setInput('');
    setImagePreview(null);
    setImageBase64(null);
  }, [input, imageBase64, loading, onSend, videoPath]);

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleImageFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      // Strip the data:...;base64, prefix — the API wants raw base64.
      const base64 = dataUrl.split(',')[1];
      setImagePreview(dataUrl);
      setImageBase64(base64);
    };
    reader.readAsDataURL(file);
  };

  const onImageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleImageFile(e.target.files[0]);
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-gradient-to-br from-gray-900/40 to-gray-950/60 rounded-2xl overflow-hidden border border-gray-800/30 shadow-2xl">

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto px-6 md:px-8 py-8 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center py-16">
            <div className="p-3 bg-cyan-900/20 rounded-2xl">
              <Sparkles className="w-10 h-10 text-cyan-400/80" />
            </div>
            <p className="text-gray-400 text-base max-w-sm leading-relaxed font-medium">
              {videoReady
                ? '✨ Ask anything about the video — find clips, answer questions, or search by image.'
                : videoPath
                ? '⏳ Video is being indexed. You can already ask general questions.'
                : '🎬 Upload a video on the left, then chat with Nova about it.'}
            </p>

            {videoReady && (
              <div className="flex flex-wrap gap-3 justify-center mt-4 pt-4 border-t border-gray-700/30">
                {SUGGESTIONS.map(s => (
                  <button
                    key={s}
                    onClick={() => onSend(s, videoPath ?? undefined)}
                    className="text-xs bg-gradient-to-r from-cyan-900/50 to-cyan-800/30 hover:from-cyan-900/80 hover:to-cyan-700/50 text-cyan-300 px-4 py-2 rounded-full border border-cyan-700/50 hover:border-cyan-600/80 transition-all duration-200 hover:shadow-lg hover:shadow-cyan-900/30 font-medium"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map(msg => <MessageBubble key={msg.id} message={msg} />)
        )}

        {/* Thinking indicator */}
        {loading && (
          <div className="flex items-center gap-3 text-cyan-400/80 text-sm pl-1 py-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="font-medium">Nova is thinking…</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-700/50 rounded-xl px-4 py-3 text-red-300 text-sm font-medium shadow-lg shadow-red-900/20">
            ⚠️ {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Image preview strip */}
      {imagePreview && (
        <div className="px-6 pb-3 flex items-center gap-3 border-t border-gray-700/30 pt-3">
          <div className="relative">
            <img
              src={imagePreview}
              alt="Query image"
              className="h-16 w-auto rounded-lg border-2 border-cyan-600/50 object-cover shadow-lg shadow-cyan-900/30"
            />
            <button
              onClick={() => { setImagePreview(null); setImageBase64(null); }}
              className="absolute -top-2 -right-2 bg-red-600 hover:bg-red-500 rounded-full p-1 transition-colors duration-200 shadow-lg"
            >
              <X className="w-3 h-3 text-white" />
            </button>
          </div>
          <p className="text-xs text-gray-400">
            🖼️ Image attached — Nova will find visually similar clips.
          </p>
        </div>
      )}

      {/* Input bar */}
      <div className="border-t border-gray-700/30 p-4 bg-gray-950/50 backdrop-blur-sm shrink-0">
        <div className="flex items-end gap-3">

          {/* Clear memory */}
          <button
            onClick={onReset}
            title="Clear conversation"
            className="p-2.5 text-gray-500 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition-all duration-200 shrink-0"
          >
            <Trash2 className="w-4 h-4" />
          </button>

          {/* Attach image */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onImageInputChange}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Attach reference image (visual search)"
            className="p-2.5 text-gray-500 hover:text-cyan-400 hover:bg-cyan-900/20 rounded-lg transition-all duration-200 shrink-0"
          >
            <ImagePlus className="w-4 h-4" />
          </button>

          {/* Textarea */}
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            rows={1}
            placeholder={
              videoReady
                ? 'Ask about the video, find a clip, or ask a question…'
                : 'Type a message…'
            }
            className="flex-1 resize-none bg-gray-800/60 hover:bg-gray-800/80 focus:bg-gray-800 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-500 border border-gray-700/50 focus:border-cyan-600/70 focus:outline-none transition-all duration-200 max-h-36 leading-relaxed"
            style={{ fieldSizing: 'content' } as React.CSSProperties}
          />

          {/* Send */}
          <button
            onClick={handleSend}
            disabled={(!input.trim() && !imageBase64) || loading}
            title="Send (Enter)"
            className="p-3 rounded-xl bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-500 hover:to-cyan-600 disabled:from-gray-700 disabled:to-gray-800 disabled:text-gray-600 text-white transition-all duration-200 shrink-0 shadow-lg shadow-cyan-900/50 hover:shadow-cyan-700/70 disabled:shadow-none"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {videoPath && !videoReady && (
          <p className="text-[11px] text-amber-600/80 mt-2 ml-11 font-medium">
            ⏳ Video is still being indexed — clip search won't work until processing completes.
          </p>
        )}
      </div>
    </div>
  );
}
