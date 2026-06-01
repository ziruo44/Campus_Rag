<script setup lang="ts">
import type { PendingPrompt } from "../types/chat";

defineProps<{
  prompt: PendingPrompt;
  disabled?: boolean;
  loading?: boolean;
}>();

const emit = defineEmits<{
  reply: [message: string];
}>();
</script>

<template>
  <section class="workflow-prompt" :class="`workflow-prompt--${prompt.kind}`">
    <div class="workflow-prompt__header">
      <p class="workflow-prompt__eyebrow">{{ prompt.eyebrow }}</p>
      <h2 class="workflow-prompt__title">{{ prompt.title }}</h2>
      <p class="workflow-prompt__description">{{ prompt.description }}</p>
    </div>

    <div class="workflow-prompt__slots">
      <div class="workflow-prompt__slot">
        <span class="workflow-prompt__slot-label">起点</span>
        <strong class="workflow-prompt__slot-value">
          {{ prompt.startLocation || "待补充" }}
        </strong>
      </div>
      <div class="workflow-prompt__slot">
        <span class="workflow-prompt__slot-label">终点</span>
        <strong class="workflow-prompt__slot-value">
          {{ prompt.endLocation || "待补充" }}
        </strong>
      </div>
    </div>

    <div v-if="prompt.suggestions.length > 0" class="workflow-prompt__actions">
      <button
        v-for="(suggestion, index) in prompt.suggestions"
        :key="suggestion"
        class="workflow-prompt__chip"
        :class="{ 'workflow-prompt__chip--primary': index === 0 }"
        type="button"
        :disabled="disabled || loading"
        @click="emit('reply', suggestion)"
      >
        {{ suggestion }}
      </button>
    </div>

    <p class="workflow-prompt__hint">{{ prompt.hint }}</p>
  </section>
</template>
