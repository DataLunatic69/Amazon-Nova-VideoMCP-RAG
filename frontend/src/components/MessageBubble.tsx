import { VideoClip } from './VideoClip';
import type { ChatMessage } from '../api/types';

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
      <div
        className={`flex flex-col gap-2.5 max-w-[80%] ${
          isUser ? 'items-end' : 'items-start'
        }`}
      >
        {/* Role label */}
        <span className="text-[10px] text-gray-500 px-2 font-semibold uppercase tracking-wide">
          {isUser ? '👤 You' : '🤖 Nova'}
        </span>

        {/* Message bubble */}
        <div
          className={`px-5 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap shadow-lg transition-all duration-200 ${
            isUser
              ? 'bg-gradient-to-br from-cyan-600 to-cyan-700 text-white rounded-br-sm shadow-cyan-900/40 hover:shadow-cyan-700/60'
              : 'bg-gradient-to-br from-gray-800 to-gray-850 text-gray-100 rounded-bl-sm border border-cyan-600/20 shadow-gray-900/40 hover:border-cyan-500/40'
          }`}
        >
          {message.content}
        </div>

        {/* Video clip (if any) */}
        {message.clip_path && (
          <div className="mt-1 animate-in fade-in duration-500">
            <VideoClip
              clipPath={message.clip_path}
              label={message.clip_path.split('/').pop()}
            />
          </div>
        )}

        {/* Timestamp */}
        <span className="text-[10px] text-gray-600 px-2 font-medium">
          {message.timestamp.toLocaleTimeString([], {
            hour:   '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}
