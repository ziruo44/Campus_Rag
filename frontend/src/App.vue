<script setup lang="ts">
import { computed, onMounted } from "vue";

import ChatComposer from "./components/ChatComposer.vue";
import HistoryPanel from "./components/HistoryPanel.vue";
import MessageList from "./components/MessageList.vue";
import StatusBanner from "./components/StatusBanner.vue";
import WorkflowPromptCard from "./components/WorkflowPromptCard.vue";
import { useChatSession } from "./composables/useChatSession";

const {
  errorMessage,
  hasMessages,
  historyItems,
  infoMessage,
  isHistoryLoading,
  isLoading,
  isRestoring,
  openThread,
  pendingPrompt,
  removeThread,
  removeTurn,
  replyToPendingPrompt,
  resetSession,
  restoreSession,
  sendMessage,
  threadId,
  threadTitle,
  visibleMessages,
} = useChatSession();

const activeConversationLabel = computed(() => threadTitle.value.trim() || "未命名会话");

const consoleTitle = computed(() =>
  hasMessages.value ? activeConversationLabel.value : "想了解温州商学院的什么？",
);

const consoleSubtitle = computed(() =>
  hasMessages.value
    ? "会话会自动保存。当前如果进入导航补充或确认阶段，输入框会继续上方流程。"
    : "围绕学院、专业、课程、培养方案或就业方向直接提问。",
);

const sessionBadge = computed(() => (threadId.value ? "已保存" : "新会话"));

const composerPlaceholder = computed(() => {
  if (pendingPrompt.value?.kind === "navigation_missing") {
    return "例如：起点是北门，或直接补充完整起终点";
  }
  if (pendingPrompt.value?.kind === "navigation_confirm") {
    return "回复“确认”，或直接修改起点/终点";
  }
  return "例如：计算机科学与技术和人工智能有什么区别？";
});

const composerHint = computed(() =>
  pendingPrompt.value
    ? "这条输入会继续上方流程，不会重新开启一轮导航理解。"
    : "Enter 发送，Shift + Enter 换行",
);

onMounted(() => {
  void restoreSession();
});
</script>

<template>
  <div class="page-shell">
    <main class="workspace workspace--copilot">
      <HistoryPanel
        :active-thread-id="threadId"
        :items="historyItems"
        :disabled="isLoading || isRestoring"
        :loading="isHistoryLoading"
        @select="openThread"
        @remove="removeThread"
      />

      <section class="console console--copilot">
        <header class="console-topbar">
          <div class="console-topbar__heading">
            <p class="console-topbar__eyebrow">WENZHOU BUSINESS COLLEGE AGENT</p>
            <div class="console-topbar__title-row">
              <h1 class="console-topbar__title">{{ consoleTitle }}</h1>
              <span class="console-topbar__badge">{{ sessionBadge }}</span>
            </div>
            <p class="console-topbar__subtitle">{{ consoleSubtitle }}</p>
          </div>

          <button
            class="console-topbar__action"
            type="button"
            :disabled="isLoading || isRestoring"
            @click="resetSession"
          >
            新对话
          </button>
        </header>

        <StatusBanner v-if="infoMessage" tone="info" :message="infoMessage" />
        <StatusBanner v-if="errorMessage" tone="error" :message="errorMessage" />

        <div class="console-body">
          <section
            class="conversation-shell"
            :class="{ 'conversation-shell--empty': !hasMessages && !isRestoring }"
          >
            <MessageList
              :messages="visibleMessages"
              :is-restoring="isRestoring"
              :disabled="isLoading || isRestoring"
              @remove-turn="removeTurn"
            />
          </section>

          <div class="composer-dock">
            <WorkflowPromptCard
              v-if="pendingPrompt"
              :prompt="pendingPrompt"
              :disabled="isRestoring"
              :loading="isLoading"
              @reply="replyToPendingPrompt"
            />

            <ChatComposer
              :disabled="isRestoring"
              :hint="composerHint"
              :loading="isLoading"
              :placeholder="composerPlaceholder"
              @submit="sendMessage"
            />
          </div>
        </div>
      </section>
    </main>
  </div>
</template>
