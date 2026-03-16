/**
 * HealthBadge — tiny pill that shows API backend liveness.
 *
 * Polls GET /health every 30 s and reflects the result:
 *   ● green  → API is reachable
 *   ● amber  → checking (first load or just reconnected)
 *   ● red    → API unreachable
 *
 * Clicking the badge triggers an immediate re-check.
 */

import { useState, useEffect, useCallback } from 'react';
import { checkHealth } from '../api/client';

type ApiStatus = 'checking' | 'ok' | 'error';

const POLL_MS = 30_000;

export function HealthBadge() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');

  const check = useCallback(async () => {
    setApiStatus('checking');
    const ok = await checkHealth();
    setApiStatus(ok ? 'ok' : 'error');
  }, []);

  useEffect(() => {
    check();
    const id = setInterval(check, POLL_MS);
    return () => clearInterval(id);
  }, [check]);

  const dot: Record<ApiStatus, string> = {
    checking: 'bg-amber-400 animate-pulse',
    ok:       'bg-emerald-400',
    error:    'bg-red-500 animate-pulse',
  };

  const label: Record<ApiStatus, string> = {
    checking: 'Checking…',
    ok:       'API online',
    error:    'API offline',
  };

  const text: Record<ApiStatus, string> = {
    checking: 'text-amber-400',
    ok:       'text-emerald-400',
    error:    'text-red-400',
  };

  return (
    <button
      onClick={check}
      title="Click to recheck API health"
      className="flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-cyan-900/20 transition-all duration-200 border border-transparent hover:border-cyan-600/30"
    >
      <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${dot[apiStatus]} shadow-lg`} />
      <span className={`text-xs font-semibold hidden sm:inline ${text[apiStatus]}`}>
        {label[apiStatus]}
      </span>
    </button>
  );
}
