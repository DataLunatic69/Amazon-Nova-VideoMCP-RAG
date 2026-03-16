// ── API request / response types — mirror app/video_rag/agent/models.py ────────

export interface UserMessageRequest {
  message: string;
  video_path?: string;
  image_base64?: string;
}

export interface AssistantMessageResponse {
  message: string;
  clip_path: string | null;
}

export interface ResetMemoryResponse {
  message: string;
}

export interface HistoryMessage {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  count: number;
  messages: HistoryMessage[];
}

// ── Video endpoints — mirror app/api/schemas.py ────────────────────────────────

export interface VideoUploadResponse {
  message: string;
  video_path: string;
}

export interface ProcessVideoResponse {
  message: string;
  task_id: string;
}

export type TaskStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'not_found';

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
}

// ── UI-only types ───────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  clip_path?: string | null;
  timestamp: Date;
}
