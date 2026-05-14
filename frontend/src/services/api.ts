import type {
  ChatRequest,
  ChatResponse,
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
  threadId?: string,
): Promise<ChatResponse> {
  const body: ChatRequest = {
    message,
    ...(threadId ? { thread_id: threadId } : {}),
  };

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });

  return parseJson<ChatResponse>(response);
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
