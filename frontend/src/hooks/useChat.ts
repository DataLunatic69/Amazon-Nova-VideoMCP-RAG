/**
 * useChat — manages the text conversation with the NovaAgent.
 *
 * Maintains a local message list; API calls are made through client.ts.
 */

import { useState, useCallback } from 'react';
import { sendChat, resetMemory as apiResetMemory } from '../api/client';
import type { ChatMessage } from '../api/types';

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string, videoPath?: string, imageBase64?: string) => {
      if (!text.trim()) return;

      const userMsg: ChatMessage = {
        id:        crypto.randomUUID(),
        role:      'user',
        content:   text,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, userMsg]);
      setLoading(true);
      setError(null);

      try {
        const res = await sendChat(text, videoPath, imageBase64);
        const assistantMsg: ChatMessage = {
          id:        crypto.randomUUID(),
          role:      'assistant',
          content:   res.message,
          clip_path: res.clip_path,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMsg]);
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const resetMemory = useCallback(async () => {
    await apiResetMemory();
    setMessages([]);
    setError(null);
  }, []);

  return { messages, loading, error, sendMessage, resetMemory };
}
