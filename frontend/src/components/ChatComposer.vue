<script setup lang="ts">
import { ref } from "vue";
import type { ComposerSubmitPayload } from "../types/chat";

const props = defineProps<{
  disabled?: boolean;
  loading?: boolean;
  hint?: string;
  placeholder?: string;
}>();

const emit = defineEmits<{
  submit: [payload: ComposerSubmitPayload];
}>();

const draft = ref("");

function submitMessage(): void {
  const normalized = draft.value.trim();
  if (!normalized || props.disabled || props.loading) {
    return;
  }

  emit("submit", {
    message: normalized,
  });
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
      rows="2"
      :disabled="disabled || loading"
      :placeholder="placeholder || '例如：计算机科学与技术和人工智能有什么区别？'"
      @keydown="handleKeydown"
    />
    <div class="composer__footer">
      <p class="composer__hint">{{ hint || "Enter 发送，Shift + Enter 换行" }}</p>
      <div class="composer__actions">
        <button class="composer__button" type="submit" :disabled="disabled || loading">
          {{ loading ? "思考中..." : "发送" }}
        </button>
      </div>
    </div>
  </form>
</template>
