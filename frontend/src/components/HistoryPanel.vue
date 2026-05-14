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

function formatTimestamp(value: string): string {
  if (!value) {
    return "";
  }

  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function handleRemove(threadId: string, event: MouseEvent): void {
  event.stopPropagation();
  emit("remove", threadId);
}
</script>

<template>
  <aside class="history-panel">
    <div class="history-panel__header">
      <div>
        <p class="history-panel__eyebrow">SESSION HISTORY</p>
        <h2 class="history-panel__title">历史会话</h2>
      </div>
      <span class="history-panel__count">{{ items.length }}</span>
    </div>

    <p v-if="loading" class="history-panel__placeholder">正在整理历史记录…</p>
    <p v-else-if="items.length === 0" class="history-panel__placeholder">
      当前还没有已保存的会话。
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
          <div class="history-panel__item-top">
            <span class="history-panel__name">{{ item.title || "New Session" }}</span>
            <span class="history-panel__turns">{{ item.turn_count }} turns</span>
          </div>
          <p class="history-panel__preview">{{ item.preview || "No preview" }}</p>
          <div class="history-panel__footer">
            <span class="history-panel__id">{{ item.thread_id }}</span>
            <time class="history-panel__time">{{ formatTimestamp(item.updated_at) }}</time>
          </div>
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
