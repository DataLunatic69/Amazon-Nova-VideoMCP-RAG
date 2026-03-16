/**
 * Thin fetch wrappers for every backend endpoint.
 *
 * Base URL is empty string so all requests go through Vite's dev proxy
 * (proxied to http://localhost:8080). In production, set VITE_API_BASE
 * in your environment and replace the empty string below.
 */

import type {
  AssistantMessageResponse,
  ChatHistoryResponse,
  ProcessVideoResponse,
  ResetMemoryResponse,
  TaskStatusResponse,
  VideoUploadResponse,
} from './types';

const BASE = import.meta.env.VITE_API_BASE ?? '';

async function _checkOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res;
}

// ── Video ──────────────────────────────────────────────────────────────────────

/** Upload a video file to SHARED_MEDIA_DIR. */
export async function uploadVideo(file: File): Promise<VideoUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await _checkOk(
    await fetch(`${BASE}/api/v1/video/upload`, { method: 'POST', body: form }),
  );
  return res.json();
}

/** Kick off the background ingestion pipeline for an already-uploaded video. */
export async function processVideo(videoPath: string): Promise<ProcessVideoResponse> {
  const res = await _checkOk(
    await fetch(`${BASE}/api/v1/video/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path: videoPath }),
    }),
  );
  return res.json();
}

/** Poll ingestion task status. */
export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const res = await _checkOk(await fetch(`${BASE}/api/v1/video/task-status/${taskId}`));
  return res.json();
}

/**
 * Build a URL that serves a media file (clip or uploaded video).
 * clip_path from the backend is an absolute filesystem path — we only
 * need the filename component to construct the URL.
 */
export function getMediaUrl(clipPath: string): string {
  const filename = clipPath.replace(/\\/g, '/').split('/').pop() ?? clipPath;
  return `${BASE}/api/v1/video/media/${encodeURIComponent(filename)}`;
}

// ── Chat ───────────────────────────────────────────────────────────────────────

/** Send a text message to the NovaAgent. */
export async function sendChat(
  message: string,
  videoPath?: string,
  imageBase64?: string,
): Promise<AssistantMessageResponse> {
  const res = await _checkOk(
    await fetch(`${BASE}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        video_path: videoPath ?? null,
        image_base64: imageBase64 ?? null,
      }),
    }),
  );
  return res.json();
}

/** Wipe the agent's conversation memory. */
export async function resetMemory(): Promise<ResetMemoryResponse> {
  const res = await _checkOk(
    await fetch(`${BASE}/api/v1/chat/memory`, { method: 'DELETE' }),
  );
  return res.json();
}

/** Fetch the last n messages from the agent's memory. */
export async function getChatHistory(n = 50): Promise<ChatHistoryResponse> {
  const res = await _checkOk(
    await fetch(`${BASE}/api/v1/chat/history?n=${n}`),
  );
  return res.json();
}

// ── Health ─────────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
