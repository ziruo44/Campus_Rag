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
  ChatStreamEvent,
  PendingPrompt,
  ThreadListItem,
  ThreadResponse,
  ThreadTurnDTO,
} from "../types/chat";

const STORAGE_KEY = "rag-agent:active-thread-id";
const NAVIGATION_CONFIRM_HEADER = "导航工具需要你确认后再执行。";
const NAVIGATION_MISSING_HEADER = "导航信息还不完整。";
const UNKNOWN_SLOT_VALUE = "待补充";

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

function normalizePromptSlot(value: string | null): string | null {
  if (!value) {
    return null;
  }

  const normalized = value.trim();
  if (!normalized || normalized === UNKNOWN_SLOT_VALUE) {
    return null;
  }

  return normalized;
}

function extractPromptSlot(content: string, label: "起点：" | "终点："): string | null {
  const line = content
    .split("\n")
    .map((item) => item.trim())
    .find((item) => item.startsWith(label));

  return normalizePromptSlot(line?.slice(label.length) ?? null);
}

function extractNumberedReplies(content: string): string[] {
  return content
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^\d+\.\s+/.test(line))
    .map((line) => line.replace(/^\d+\.\s+/, "").trim())
    .filter(Boolean);
}

function parsePendingPrompt(message: ChatMessage): PendingPrompt | null {
  if (message.role !== "assistant") {
    return null;
  }

  const content = message.content.trim();
  if (!content) {
    return null;
  }

  const startLocation = extractPromptSlot(content, "起点：");
  const endLocation = extractPromptSlot(content, "终点：");

  if (content.startsWith(NAVIGATION_MISSING_HEADER)) {
    const suggestions = extractNumberedReplies(content).slice(0, 3);
    const remembered = [
      startLocation ? `已记住起点 ${startLocation}` : "",
      endLocation ? `已记住终点 ${endLocation}` : "",
    ].filter(Boolean);

    return {
      id: message.id,
      kind: "navigation_missing",
      eyebrow: "ACTION REQUIRED",
      title:
        startLocation && !endLocation
          ? "补充终点后继续导航"
          : !startLocation && endLocation
            ? "补充起点后继续导航"
            : "补全导航信息",
      description: remembered.length
        ? `${remembered.join("，")}。直接补全缺失字段后，我会继续当前导航流程。`
        : "直接补全起点和终点，我会继续当前导航流程。",
      startLocation,
      endLocation,
      hint: "补全后会先进入确认，再执行导航。",
      suggestions,
    };
  }

  if (content.startsWith(NAVIGATION_CONFIRM_HEADER)) {
    return {
      id: message.id,
      kind: "navigation_confirm",
      eyebrow: "READY TO RUN",
      title: "确认并执行导航",
      description:
        startLocation && endLocation
          ? `已记住起点 ${startLocation} 和终点 ${endLocation}。确认后会直接执行，不再重新理解目的地。`
          : "导航信息已补全，确认后会直接执行。",
      startLocation,
      endLocation,
      hint: "也可以直接在下方输入：起点改为南门 / 终点改为图书馆。",
      suggestions: ["确认", "取消"],
    };
  }

  return null;
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
  const pendingPrompt = computed<PendingPrompt | null>(() => {
    const lastMessage = messages.value[messages.value.length - 1];
    if (!lastMessage) {
      return null;
    }
    return parsePendingPrompt(lastMessage);
  });
  const visibleMessages = computed(() =>
    pendingPrompt.value
      ? messages.value.filter((message) => message.id !== pendingPrompt.value?.id)
      : messages.value,
  );

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
        (event: ChatStreamEvent) => {
          resolvedThreadId = event.thread_id;
          threadId.value = event.thread_id;
          writeStoredThreadId(event.thread_id);

          messages.value = updateMessage(
            updateMessage(messages.value, optimisticUserMessage.id, (message) => ({
              ...message,
              turnId: event.turn_id,
            })),
            streamingAssistantMessage.id,
            (message) => ({
              ...message,
              turnId: event.turn_id,
            }),
          );

          if (event.event === "delta" && event.content) {
            messages.value = updateMessage(
              messages.value,
              streamingAssistantMessage.id,
              (message) => ({
                ...message,
                content: `${message.content}${event.content}`,
              }),
            );
          }

          if (event.event === "error") {
            streamError = event.error;
          }
        },
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

  async function replyToPendingPrompt(message: string): Promise<void> {
    await sendMessage({ message });
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
    pendingPrompt,
    refreshHistory,
    removeThread,
    removeTurn,
    replyToPendingPrompt,
    resetSession,
    restoreSession,
    sendMessage,
    threadId,
    threadTitle,
    visibleMessages,
  };
}
