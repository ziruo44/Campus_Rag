<script setup lang="ts">
defineProps<{
  threadId: string | null;
  hasMessages: boolean;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  reset: [];
}>();
</script>

<template>
  <header class="chat-header">
    <div class="chat-header__eyebrow">WENZHOU BUSINESS COLLEGE</div>
    <div class="chat-header__main">
      <div>
        <h1 class="chat-header__title">Campus Counselor Board</h1>
        <p class="chat-header__subtitle">
          面向学院与专业介绍的多轮咨询界面，保留当前会话的完整历史记录。
        </p>
      </div>

      <button
        class="chat-header__reset"
        type="button"
        :disabled="disabled"
        @click="emit('reset')"
      >
        新建会话
      </button>
    </div>

    <div class="chat-header__meta">
      <span class="chat-header__label">咨询编号</span>
      <span class="chat-header__thread">
        {{ threadId || "尚未创建" }}
      </span>
      <span class="chat-header__history">
        {{ hasMessages ? "当前会话历史已保留" : "等待第一条提问" }}
      </span>
    </div>
  </header>
</template>
