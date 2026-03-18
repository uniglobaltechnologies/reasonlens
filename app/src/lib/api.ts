const API_BASE = import.meta.env.VITE_API_URL || "https://reasonlens-api.azurewebsites.net/api";

function getToken(): string | null {
  return localStorage.getItem("reasonlens_token");
}

export function setToken(token: string) {
  localStorage.setItem("reasonlens_token", token);
}

export function clearToken() {
  localStorage.removeItem("reasonlens_token");
}

export function isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    if (!payload.exp) return false;
    return Date.now() >= payload.exp * 1000;
  } catch {
    return true;
  }
}

export function isAuthenticated(): boolean {
  const token = getToken();
  if (!token) return false;
  if (isTokenExpired(token)) {
    clearToken();
    return false;
  }
  return true;
}

async function request<T = any>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
    });
  } catch {
    throw new ApiError(
      "Failed to reach API (network/CORS). Check custom domain CORS and API URL.",
      0
    );
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    const message = body.error || res.statusText;
    if (res.status === 401) {
      clearToken();
      throw new ApiError(`${message}. Please sign in.`, res.status);
    }
    throw new ApiError(message, res.status);
  }

  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(
      text ? `Unexpected response: ${text.slice(0, 200)}` : "Empty response from server",
      res.status
    );
  }
}

export async function apiGet<T = any>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

export async function apiPost<T = any>(path: string, body?: any): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body ? JSON.stringify(body) : undefined,
  });
}

export async function apiDelete<T = any>(path: string, body?: any): Promise<T> {
  return request<T>(path, {
    method: "DELETE",
    body: body ? JSON.stringify(body) : undefined,
  });
}

// SSE streaming helper
export async function apiStream(
  path: string,
  body: any,
  onChunk: (text: string) => void,
  onDone?: () => void,
  onError?: (err: string) => void
): Promise<void> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  } catch {
    onError?.(
      "Failed to reach API (network/CORS). Check custom domain CORS and API URL."
    );
    return;
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    const message = err.error || res.statusText;
    onError?.(res.status === 401 ? `${message}. Please sign in.` : message);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6).trim();
        if (data === "[DONE]") {
          onDone?.();
          return;
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.content) onChunk(parsed.content);
          if (parsed.error) onError?.(parsed.error);
        } catch (parseErr) {
          console.warn("SSE parse error:", parseErr, "data:", data);
        }
      }
    }
  }

  // Process any remaining data in the buffer
  if (buffer.trim()) {
    if (buffer.startsWith("data: ")) {
      const data = buffer.slice(6).trim();
      if (data !== "[DONE]") {
        try {
          const parsed = JSON.parse(data);
          if (parsed.content) onChunk(parsed.content);
          if (parsed.error) onError?.(parsed.error);
        } catch {
          // ignore trailing partial data
        }
      }
    }
  }

  onDone?.();
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}
