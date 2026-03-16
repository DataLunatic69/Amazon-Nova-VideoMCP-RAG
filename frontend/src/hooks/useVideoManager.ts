/**
 * useVideoManager — upload → process → poll status.
 *
 * Drives the full video ingestion pipeline:
 *  1. User drops / picks a file.
 *  2. POST /api/v1/video/upload  →  server-side path.
 *  3. POST /api/v1/video/process →  task_id.
 *  4. Poll  GET /api/v1/video/task-status every 3 s until completed/failed.
 */

import { useState, useCallback, useRef } from 'react';
import { uploadVideo, processVideo, getTaskStatus } from '../api/client';

export type IngestStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

export interface VideoState {
  videoPath: string | null;
  fileName: string | null;
  ingestStatus: IngestStatus;
  statusMessage: string;
  taskId: string | null;
}

const POLL_INTERVAL_MS = 3000;

export function useVideoManager() {
  const [state, setState] = useState<VideoState>({
    videoPath: null,
    fileName: null,
    ingestStatus: 'idle',
    statusMessage: '',
    taskId: null,
  });

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (taskId: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const { status } = await getTaskStatus(taskId);
          if (status === 'completed') {
            stopPolling();
            setState(s => ({
              ...s,
              ingestStatus: 'completed',
              statusMessage: 'Ready — start chatting!',
            }));
          } else if (status === 'failed') {
            stopPolling();
            setState(s => ({
              ...s,
              ingestStatus: 'failed',
              statusMessage: 'Ingestion failed. Try re-uploading.',
            }));
          } else {
            const label: Record<string, string> = {
              pending:     'Queued…',
              in_progress: 'Processing…',
              not_found:   'Task not found',
            };
            setState(s => ({
              ...s,
              statusMessage: label[status] ?? status,
            }));
          }
        } catch {
          // Swallow transient network errors during polling.
        }
      }, POLL_INTERVAL_MS);
    },
    [stopPolling],
  );

  const handleFile = useCallback(
    async (file: File) => {
      stopPolling();
      setState({
        videoPath: null,
        fileName: file.name,
        ingestStatus: 'uploading',
        statusMessage: 'Uploading…',
        taskId: null,
      });

      try {
        const { video_path } = await uploadVideo(file);

        setState(s => ({
          ...s,
          videoPath: video_path,
          ingestStatus: 'processing',
          statusMessage: 'Starting ingestion…',
        }));

        const { task_id } = await processVideo(video_path);

        setState(s => ({ ...s, taskId: task_id, statusMessage: 'Processing…' }));
        startPolling(task_id);
      } catch (err) {
        setState(s => ({
          ...s,
          ingestStatus: 'failed',
          statusMessage: String(err),
        }));
      }
    },
    [stopPolling, startPolling],
  );

  return { ...state, handleFile };
}
