export type ChatRequest = {
  message: string;
  thread_id?: string;
  precise_mode?: boolean;
};

export type ChatResponse = {
  thread_id: string;
  answer: string;
};

export type ChatStreamChunk = {
  content: string;
  is_final: boolean;
  thread_id?: string;
  turn_id?: string;
  error?: string;
};

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
  preciseMode: boolean;
};
