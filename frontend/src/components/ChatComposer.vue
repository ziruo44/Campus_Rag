<script setup lang="ts">
import { ref } from "vue";
import type { ComposerSubmitPayload } from "../types/chat";

const props = defineProps<{
  disabled?: boolean;
  loading?: boolean;
}>();

const emit = defineEmits<{
  submit: [payload: ComposerSubmitPayload];
}>();

const draft = ref("");
const preciseMode = ref(false);

function submitMessage(): void {
  const normalized = draft.value.trim();
  if (!normalized || props.disabled || props.loading) {
    return;
  }

  emit("submit", {
    message: normalized,
    preciseMode: preciseMode.value,
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
      placeholder="例如：计算机科学与技术和人工智能有什么区别？"
      @keydown="handleKeydown"
    />
    <div class="composer__footer">
      <p class="composer__hint">Enter 发送，Shift + Enter 换行</p>
      <div class="composer__actions">
        <label class="composer__toggle composer__toggle--compact">
          <input
            v-model="preciseMode"
            class="composer__toggle-input"
            type="checkbox"
            :disabled="disabled || loading"
          />
          <span class="composer__toggle-switch" aria-hidden="true"></span>
          <span class="composer__toggle-copy">
            <span class="composer__toggle-title">精准回复</span>
            <span class="composer__toggle-note">更慢</span>
          </span>
        </label>
        <button class="composer__button" type="submit" :disabled="disabled || loading">
          {{ loading ? "思考中..." : "发送" }}
        </button>
      </div>
    </div>
  </form>
</template>
