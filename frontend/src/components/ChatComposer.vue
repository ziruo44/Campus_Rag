<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{
  disabled?: boolean;
  loading?: boolean;
}>();

const emit = defineEmits<{
  submit: [message: string];
}>();

const draft = ref("");

function submitMessage(): void {
  const normalized = draft.value.trim();
  if (!normalized || props.disabled || props.loading) {
    return;
  }

  emit("submit", normalized);
  draft.value = "";
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage();
  }
}
</script>

<template>
  <form class="composer" @submit.prevent="submitMessage">
    <label class="composer__label" for="chat-input">输入问题</label>
    <textarea
      id="chat-input"
      v-model="draft"
      class="composer__input"
      rows="4"
      :disabled="disabled || loading"
      placeholder="例如：计算机科学与技术和人工智能有什么区别？"
      @keydown="handleKeydown"
    />
    <div class="composer__footer">
      <p class="composer__hint">按 Enter 发送，Shift + Enter 换行。</p>
      <button class="composer__button" type="submit" :disabled="disabled || loading">
        {{ loading ? "发送中…" : "发送问题" }}
      </button>
    </div>
  </form>
</template>
