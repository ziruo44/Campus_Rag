<script setup lang="ts">
import type { ChatMessage } from "../types/chat";

defineProps<{
  messages: ChatMessage[];
  isRestoring: boolean;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  removeTurn: [turnId: string];
}>();

function handleRemoveTurn(turnId: string, event: MouseEvent): void {
  event.stopPropagation();
  emit("removeTurn", turnId);
}

function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}
</script>

<template>
  <section class="message-board" aria-live="polite">
    <div v-if="isRestoring" class="message-board__empty">
      <p class="message-board__empty-kicker">RESTORING SESSION</p>
      <p class="message-board__empty-title">正在恢复会话记录</p>
      <p class="message-board__empty-copy">
        如果本地保存了历史线程，页面会自动拉回完整上下文。
      </p>
    </div>

    <div v-else-if="messages.length === 0" class="message-board__empty">
      <p class="message-board__empty-kicker">CAMPUS KNOWLEDGE AGENT</p>
      <p class="message-board__empty-title">从一个问题开始</p>
      <p class="message-board__empty-copy">
        这里会保留当前会话内的全部问答，适合连续追问学院、专业、课程与培养方向。
      </p>
    </div>

    <ol v-else class="message-board__list">
      <li
        v-for="message in messages"
        :key="message.id"
        class="message-row"
        :class="`message-row--${message.role}`"
      >
        <div class="message-row__meta">
          <div class="message-row__identity">
            <span class="message-row__role">
              {{ message.role === "user" ? "我" : "校园知识助手" }}
            </span>
            <span v-if="message.timestamp" class="message-row__timestamp">
              {{ formatTimestamp(message.timestamp) }}
            </span>
          </div>
          <button
            v-if="message.role === 'user'"
            class="message-row__delete"
            type="button"
            :disabled="disabled"
            @click="handleRemoveTurn(message.turnId, $event)"
          >
            删除
          </button>
        </div>
        <div class="message-row__bubble">
          <p class="message-row__content">{{ message.content }}</p>
        </div>
      </li>
    </ol>
  </section>
</template>
