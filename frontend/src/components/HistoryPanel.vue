<script setup lang="ts">
import type { ThreadListItem } from "../types/chat";

defineProps<{
  activeThreadId: string | null;
  items: ThreadListItem[];
  disabled?: boolean;
  loading?: boolean;
}>();

const emit = defineEmits<{
  select: [threadId: string];
  remove: [threadId: string];
}>();

function handleRemove(threadId: string, event: MouseEvent): void {
  event.stopPropagation();
  emit("remove", threadId);
}

function formatTitle(title: string): string {
  const normalized = title.trim();
  return normalized || "未命名会话";
}
</script>

<template>
  <aside class="history-panel">
    <div class="history-panel__header">
      <div>
        <p class="history-panel__eyebrow">THREADS</p>
        <h2 class="history-panel__title">历史会话</h2>
      </div>
      <span class="history-panel__count">{{ items.length }}</span>
    </div>

    <p v-if="loading" class="history-panel__placeholder">正在整理历史会话...</p>
    <p v-else-if="items.length === 0" class="history-panel__placeholder">
      发送第一条消息后，这里会自动生成会话标题。
    </p>

    <ol v-else class="history-panel__list">
      <li
        v-for="item in items"
        :key="item.thread_id"
        class="history-panel__row"
        :class="{ 'history-panel__row--active': item.thread_id === activeThreadId }"
      >
        <button
          class="history-panel__item"
          type="button"
          :disabled="disabled"
          @click="emit('select', item.thread_id)"
        >
          <span class="history-panel__name">{{ formatTitle(item.title) }}</span>
        </button>
        <button
          class="history-panel__delete"
          type="button"
          :disabled="disabled"
          @click="handleRemove(item.thread_id, $event)"
        >
          删除
        </button>
      </li>
    </ol>
  </aside>
</template>
