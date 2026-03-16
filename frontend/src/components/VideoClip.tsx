import { getMediaUrl } from '../api/client';
import { Play } from 'lucide-react';

interface VideoClipProps {
  clipPath: string;
  label?: string;
}

export function VideoClip({ clipPath, label }: VideoClipProps) {
  const url = getMediaUrl(clipPath);

  return (
    <div className="rounded-xl overflow-hidden border-2 border-cyan-500/40 bg-black/80 max-w-sm w-full shadow-2xl shadow-cyan-900/40 hover:border-cyan-400/60 transition-all duration-300 group">
      {label && (
        <p className="text-xs text-gray-400 px-3 pt-2 font-semibold truncate flex items-center gap-1">
          <Play className="w-3 h-3 text-cyan-400" />
          {label}
        </p>
      )}
      <video
        src={url}
        controls
        autoPlay
        muted
        playsInline
        className="w-full max-h-64 object-contain bg-black hover:brightness-110 transition-all duration-200"
        preload="auto"
      />
    </div>
  );
}
