import type {
  ChatRequest,
  ChatResponse,
  ChatStreamChunk,
  ThreadListItem,
  ThreadResponse,
} from "../types/chat";

const JSON_HEADERS = {
  "Content-Type": "application/json",
};

async function parseJson<T>(response: Response): Promise<T> {
  const payload = (await response.json().catch(() => null)) as
    | T
    | { detail?: string }
    | null;

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? payload.detail
        : undefined;

    const error = new Error(detail || "Request failed.");
    (error as Error & { status?: number }).status = response.status;
    throw error;
  }

  return payload as T;
}

export async function sendChat(
  message: string,
  preciseMode = false,
  threadId?: string,
): Promise<ChatResponse> {
  const body: ChatRequest = {
    message,
    precise_mode: preciseMode,
    ...(threadId ? { thread_id: threadId } : {}),
  };

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });

  return parseJson<ChatResponse>(response);
}

function parseSseChunk(rawEvent: string): ChatStreamChunk | null {
  const dataLines = rawEvent
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());

  if (dataLines.length === 0) {
    return null;
  }

  return JSON.parse(dataLines.join("\n")) as ChatStreamChunk;
}

export async function sendChatStream(
  message: string,
  onChunk: (chunk: ChatStreamChunk) => void,
  preciseMode = false,
  threadId?: string,
): Promise<void> {
  const body: ChatRequest = {
    message,
    precise_mode: preciseMode,
    ...(threadId ? { thread_id: threadId } : {}),
  };

  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    await parseJson<never>(response);
    return;
  }

  if (!response.body) {
    throw new Error("Streaming response body is unavailable.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (rawEvent) {
        const chunk = parseSseChunk(rawEvent);
        if (chunk) {
          onChunk(chunk);
        }
      }
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  const trailingEvent = buffer.trim();
  if (trailingEvent) {
    const chunk = parseSseChunk(trailingEvent);
    if (chunk) {
      onChunk(chunk);
    }
  }
}

export async function fetchThread(threadId: string): Promise<ThreadResponse> {
  const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`, {
    method: "GET",
  });

  return parseJson<ThreadResponse>(response);
}

export async function fetchThreads(): Promise<ThreadListItem[]> {
  const response = await fetch("/api/threads", {
    method: "GET",
  });

  return parseJson<ThreadListItem[]>(response);
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`, {
    method: "DELETE",
  });
  await parseJson<void>(response);
}

export async function deleteTurn(
  threadId: string,
  turnId: string,
): Promise<ThreadResponse> {
  const response = await fetch(
    `/api/threads/${encodeURIComponent(threadId)}/turns/${encodeURIComponent(turnId)}`,
    {
      method: "DELETE",
    },
  );

  return parseJson<ThreadResponse>(response);
}

export async function checkHealth(): Promise<boolean> {
  const response = await fetch("/health", { method: "GET" });
  return response.ok;
}
