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

const samplePrompts = [
  "计算机科学与技术和人工智能有什么区别？",
  "信息工程学院有哪些专业？",
  "那人工智能的核心课程有哪些？",
];

function handleRemoveTurn(turnId: string, event: MouseEvent): void {
  event.stopPropagation();
  emit("removeTurn", turnId);
}
</script>

<template>
  <section class="message-board" aria-live="polite">
    <div v-if="isRestoring" class="message-board__empty">
      <p class="message-board__empty-title">正在恢复会话记录…</p>
      <p class="message-board__empty-copy">如果本地存在咨询编号，页面会自动还原整段对话。</p>
    </div>

    <div v-else-if="messages.length === 0" class="message-board__empty">
      <p class="message-board__empty-title">开始你的第一轮咨询</p>
      <p class="message-board__empty-copy">
        这里会保留当前会话内的所有问答，刷新页面后也可以继续。
      </p>
      <ul class="message-board__prompts">
        <li v-for="prompt in samplePrompts" :key="prompt">{{ prompt }}</li>
      </ul>
    </div>

    <ol v-else class="message-board__list">
      <li
        v-for="message in messages"
        :key="message.id"
        class="message-card"
        :class="`message-card--${message.role}`"
      >
        <div class="message-card__meta">
          <span class="message-card__role">
            {{ message.role === "user" ? "用户提问" : "Agent 回答" }}
          </span>
          <span v-if="message.timestamp" class="message-card__timestamp">
            {{ new Date(message.timestamp).toLocaleString("zh-CN", { hour12: false }) }}
          </span>
          <button
            v-if="message.role === 'user'"
            class="message-card__delete"
            type="button"
            :disabled="disabled"
            @click="handleRemoveTurn(message.turnId, $event)"
          >
            删除该轮
          </button>
        </div>
        <p class="message-card__content">{{ message.content }}</p>
      </li>
    </ol>
  </section>
</template>
