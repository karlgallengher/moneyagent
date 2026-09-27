export type DomainInfo = {
  domain: string;
  name: string;
  index_dir: string;
  catalog_exists: boolean;
  page_index_exists: boolean;
  doc_count: number;
  page_count: number;
  upload_count: number;
  status: "ready" | "needs_build";
  last_built_at: string;
};

export type DomainFile = {
  filename: string;
  doc_id: string;
  size_bytes: number;
  updated_at: string;
};

export type Evidence = {
  doc_id?: string;
  page_id?: string;
  evidence_id?: string;
  heading?: string;
  quote?: string;
  score?: number;
};

export type Fact = {
  slot: string;
  value: string;
  evidence_ids: string[];
  source: "document" | "user_query";
};

export type ChatResponse = {
  query: string;
  qid: string;
  status: string;
  answer: string;
  domain: string;
  doc_ids: string[];
  evidence: Evidence[];
  facts: Fact[];
  question_info: Fact[];
  reason: string;
  confidence: number | null;
  token_usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  debug_files: Record<string, string>;
  error_type?: string;
  error_message?: string;
};

export type SessionInfo = {
  session_id: string;
  title: string;
  active_domain: string;
  turn_count: number;
  created_at: string;
  updated_at: string;
};

type ApiEnvelope<T> = {
  ok: boolean;
  data: T | null;
  error: { type: string; message: string } | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const envelope = (await response.json()) as ApiEnvelope<T>;
  if (!envelope.ok || !envelope.data) {
    throw new Error(envelope.error?.message || "MoneyAgent API returned an error.");
  }
  return envelope.data;
}

export function fetchDomains() {
  return request<{ domains: DomainInfo[] }>("/api/domains");
}

export function createDomain(payload: { domain: string; name: string; split_mode: string }) {
  return request<{ domain: string; name: string }>("/api/domains", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function uploadDomainFile(domain: string, file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<{ domain: string; filename: string }>(`/api/domains/${encodeURIComponent(domain)}/files`, {
    method: "POST",
    body
  });
}

export function replaceDomainFile(domain: string, filename: string, file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<{ domain: string; filename: string; replaced: boolean }>(
    `/api/domains/${encodeURIComponent(domain)}/files/${encodeURIComponent(filename)}`,
    { method: "PUT", body }
  );
}

export function fetchDomainFiles(domain: string) {
  return request<{ domain: string; files: DomainFile[] }>(
    `/api/domains/${encodeURIComponent(domain)}/files`
  );
}

export function deleteDomainFile(domain: string, filename: string) {
  return request<{ domain: string; filename: string; deleted: boolean }>(
    `/api/domains/${encodeURIComponent(domain)}/files/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
}

export function deleteDomain(domain: string) {
  return request<{ domain: string; deleted: boolean }>(
    `/api/domains/${encodeURIComponent(domain)}`,
    { method: "DELETE" }
  );
}

export function buildDomain(domain: string) {
  return request<{ domain: string; doc_count: number; page_count: number }>(
    `/api/domains/${encodeURIComponent(domain)}/build`,
    { method: "POST" }
  );
}

export function askMoneyAgent(payload: {
  query: string;
  domain: string;
  session_id: string;
  preferred_doc_ids: string[];
}) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export type SessionHistory = {
  session_id: string;
  title?: string;
  turn_count: number;
  active_domain: string;
  active_doc_ids: string[];
  turns: Array<{
    turn_index: number;
    user_query: string;
    assistant_answer: string;
    domain: string;
    doc_ids: string;
    created_at: string;
  }>;
};

export function fetchSession(sessionId: string) {
  return request<SessionHistory>(`/api/sessions/${encodeURIComponent(sessionId)}`);
}

export function fetchSessions() {
  return request<{ sessions: SessionInfo[] }>("/api/sessions");
}

export function createSession(title = "") {
  return request<{ session_id: string; title: string }>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title })
  });
}

export function renameSession(sessionId: string, title: string) {
  return request<{ session_id: string; title: string }>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify({ title })
  });
}

export function deleteSession(sessionId: string) {
  return request<{ session_id: string; deleted: boolean }>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE"
  });
}

export function resetSession(sessionId: string) {
  return request<{ session_id: string; reset: boolean }>("/api/sessions/reset", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId })
  });
}
