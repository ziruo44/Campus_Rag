export type ChatRequest = {
  message: string;
  thread_id?: string;
};

export type ChatResponse = {
  thread_id: string;
  answer: string;
};

type ChatStreamEventBase = {
  thread_id: string;
  turn_id: string;
};

export type ChatStreamStartEvent = ChatStreamEventBase & {
  event: "start";
};

export type ChatStreamDeltaEvent = ChatStreamEventBase & {
  event: "delta";
  content: string;
};

export type ChatStreamDoneEvent = ChatStreamEventBase & {
  event: "done";
  content?: string;
};

export type ChatStreamErrorEvent = ChatStreamEventBase & {
  event: "error";
  error: string;
  content?: string;
};

export type ChatStreamEvent =
  | ChatStreamStartEvent
  | ChatStreamDeltaEvent
  | ChatStreamDoneEvent
  | ChatStreamErrorEvent;

export type ThreadMessageDTO = {
  role: string;
  content: string;
  timestamp: string;
};

export type ThreadTurnDTO = {
  turn_id: string;
  state: string;
  user_message: ThreadMessageDTO | null;
  assistant_message: ThreadMessageDTO | null;
  updated_at: string;
};

export type ThreadResponse = {
  thread_id: string;
  title: string;
  summary: string;
  profile: Record<string, unknown>;
  turns: ThreadTurnDTO[];
};

export type ThreadListItem = {
  thread_id: string;
  title: string;
  summary: string;
  updated_at: string;
  turn_count: number;
  preview: string;
};

export type ChatMessage = {
  id: string;
  turnId: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
};

export type ComposerSubmitPayload = {
  message: string;
};

export type PendingPromptKind = "navigation_missing" | "navigation_confirm";

export type PendingPrompt = {
  id: string;
  kind: PendingPromptKind;
  eyebrow: string;
  title: string;
  description: string;
  startLocation: string | null;
  endLocation: string | null;
  hint: string;
  suggestions: string[];
};
