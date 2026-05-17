import { computed, ref } from "vue";

import {
  deleteThread,
  deleteTurn,
  fetchThread,
  fetchThreads,
  sendChatStream,
} from "../services/api";
import type {
  ChatMessage,
  ComposerSubmitPayload,
  ThreadListItem,
  ThreadResponse,
  ThreadTurnDTO,
} from "../types/chat";

const STORAGE_KEY = "rag-agent:active-thread-id";

function buildMessageId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function mapTurnsToMessages(turns: ThreadTurnDTO[]): ChatMessage[] {
  const mapped: ChatMessage[] = [];

  for (const turn of turns) {
    if (turn.user_message?.content) {
      mapped.push({
        id: `${turn.turn_id}-user`,
        turnId: turn.turn_id,
        role: "user",
        content: turn.user_message.content,
        timestamp: turn.user_message.timestamp,
      });
    }

    if (turn.assistant_message?.content) {
      mapped.push({
        id: `${turn.turn_id}-assistant`,
        turnId: turn.turn_id,
        role: "assistant",
        content: turn.assistant_message.content,
        timestamp: turn.assistant_message.timestamp,
      });
    }
  }

  return mapped;
}

function updateMessage(
  items: ChatMessage[],
  messageId: string,
  updater: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  const index = items.findIndex((message) => message.id === messageId);
  if (index === -1) {
    return items;
  }

  const nextItems = [...items];
  nextItems[index] = updater(nextItems[index]);
  return nextItems;
}

function readStoredThreadId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  const value = window.localStorage.getItem(STORAGE_KEY);
  return value?.trim() || null;
}

function writeStoredThreadId(threadId: string | null): void {
  if (typeof window === "undefined") {
    return;
  }

  if (threadId) {
    window.localStorage.setItem(STORAGE_KEY, threadId);
    return;
  }

  window.localStorage.removeItem(STORAGE_KEY);
}

function toUserFacingError(error: unknown): string {
  const status = (error as { status?: number } | undefined)?.status;

  if (status === 404) {
    return "Current session was not found. The active view has been reset.";
  }

  if (status === 500) {
    return "The backend failed to process this request. Try again shortly.";
  }

  if (status === 503) {
    return "The backend is unavailable right now. Check whether FastAPI is running.";
  }

  if (error instanceof TypeError) {
    return "The browser could not reach the backend service.";
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  return "An unexpected error occurred.";
}

export function useChatSession() {
  const threadId = ref<string | null>(null);
  const threadTitle = ref("New Session");
  const messages = ref<ChatMessage[]>([]);
  const historyItems = ref<ThreadListItem[]>([]);
  const isLoading = ref(false);
  const isRestoring = ref(true);
  const isHistoryLoading = ref(true);
  const errorMessage = ref("");
  const infoMessage = ref("");

  const hasMessages = computed(() => messages.value.length > 0);

  function applyThread(thread: ThreadResponse): void {
    threadId.value = thread.thread_id;
    threadTitle.value = thread.title || "New Session";
    messages.value = mapTurnsToMessages(thread.turns);
    writeStoredThreadId(thread.thread_id);
  }

  async function refreshHistory(): Promise<void> {
    isHistoryLoading.value = true;
    try {
      historyItems.value = await fetchThreads();
    } catch (error) {
      infoMessage.value = toUserFacingError(error);
    } finally {
      isHistoryLoading.value = false;
    }
  }

  async function restoreSession(): Promise<void> {
    errorMessage.value = "";
    infoMessage.value = "";
    isRestoring.value = true;

    await refreshHistory();

    const storedThreadId = readStoredThreadId();
    if (!storedThreadId) {
      isRestoring.value = false;
      return;
    }

    try {
      const thread = await fetchThread(storedThreadId);
      applyThread(thread);
    } catch (error) {
      threadId.value = null;
      threadTitle.value = "New Session";
      messages.value = [];
      writeStoredThreadId(null);
      infoMessage.value = toUserFacingError(error);
    } finally {
      isRestoring.value = false;
    }
  }

  async function openThread(nextThreadId: string): Promise<void> {
    const normalized = nextThreadId.trim();
    if (!normalized || normalized === threadId.value || isLoading.value) {
      return;
    }

    errorMessage.value = "";
    infoMessage.value = "";
    isRestoring.value = true;

    try {
      const thread = await fetchThread(normalized);
      applyThread(thread);
    } catch (error) {
      errorMessage.value = toUserFacingError(error);
    } finally {
      isRestoring.value = false;
    }
  }

  async function sendMessage(payload: ComposerSubmitPayload): Promise<void> {
    const normalized = payload.message.trim();
    if (!normalized || isLoading.value) {
      return;
    }

    errorMessage.value = "";
    infoMessage.value = "";
    isLoading.value = true;

    const optimisticUserMessage: ChatMessage = {
      id: buildMessageId("user"),
      turnId: buildMessageId("turn"),
      role: "user",
      content: normalized,
      timestamp: new Date().toISOString(),
    };
    messages.value.push(optimisticUserMessage);

    const streamingAssistantMessage: ChatMessage = {
      id: buildMessageId("assistant"),
      turnId: optimisticUserMessage.turnId,
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
    };
    messages.value.push(streamingAssistantMessage);

    let resolvedThreadId = threadId.value;
    let streamError = "";

    try {
      await sendChatStream(
        normalized,
        (chunk) => {
          if (chunk.thread_id) {
            resolvedThreadId = chunk.thread_id;
            threadId.value = chunk.thread_id;
            writeStoredThreadId(chunk.thread_id);
          }

          if (chunk.turn_id) {
            messages.value = updateMessage(
              updateMessage(messages.value, optimisticUserMessage.id, (message) => ({
                ...message,
                turnId: chunk.turn_id ?? message.turnId,
              })),
              streamingAssistantMessage.id,
              (message) => ({
                ...message,
                turnId: chunk.turn_id ?? message.turnId,
              }),
            );
          }

          if (chunk.content) {
            messages.value = updateMessage(
              messages.value,
              streamingAssistantMessage.id,
              (message) => ({
                ...message,
                content: `${message.content}${chunk.content}`,
              }),
            );
          }

          if (chunk.error) {
            streamError = chunk.error;
          }
        },
        payload.preciseMode,
        threadId.value ?? undefined,
      );

      if (streamError) {
        throw new Error(streamError);
      }

      if (!resolvedThreadId) {
        throw new Error("Streaming response did not include a thread ID.");
      }

      const thread = await fetchThread(resolvedThreadId);
      applyThread(thread);
      await refreshHistory();
    } catch (error) {
      if (resolvedThreadId) {
        try {
          const thread = await fetchThread(resolvedThreadId);
          applyThread(thread);
          await refreshHistory();
        } catch {
          messages.value = messages.value.filter(
            (message) =>
              message.id !== optimisticUserMessage.id &&
              message.id !== streamingAssistantMessage.id,
          );
        }
      } else {
        messages.value = messages.value.filter(
          (message) =>
            message.id !== optimisticUserMessage.id &&
            message.id !== streamingAssistantMessage.id,
        );
      }

      errorMessage.value = toUserFacingError(error);
    } finally {
      isLoading.value = false;
    }
  }

  async function removeThread(targetThreadId: string): Promise<void> {
    if (isLoading.value || isRestoring.value) {
      return;
    }

    errorMessage.value = "";
    infoMessage.value = "";
    isLoading.value = true;

    try {
      await deleteThread(targetThreadId);
      if (threadId.value === targetThreadId) {
        threadId.value = null;
        threadTitle.value = "New Session";
        messages.value = [];
        writeStoredThreadId(null);
      }
      await refreshHistory();
    } catch (error) {
      errorMessage.value = toUserFacingError(error);
    } finally {
      isLoading.value = false;
    }
  }

  async function removeTurn(turnIdToDelete: string): Promise<void> {
    if (!threadId.value || isLoading.value || isRestoring.value) {
      return;
    }

    errorMessage.value = "";
    infoMessage.value = "";
    isLoading.value = true;

    try {
      const thread = await deleteTurn(threadId.value, turnIdToDelete);
      applyThread(thread);
      await refreshHistory();
    } catch (error) {
      errorMessage.value = toUserFacingError(error);
    } finally {
      isLoading.value = false;
    }
  }

  function resetSession(): void {
    threadId.value = null;
    threadTitle.value = "New Session";
    messages.value = [];
    errorMessage.value = "";
    infoMessage.value = "Started a blank session. The next question will create a new thread.";
    writeStoredThreadId(null);
  }

  return {
    errorMessage,
    hasMessages,
    historyItems,
    infoMessage,
    isHistoryLoading,
    isLoading,
    isRestoring,
    messages,
    openThread,
    refreshHistory,
    removeThread,
    removeTurn,
    resetSession,
    restoreSession,
    sendMessage,
    threadId,
    threadTitle,
  };
}
