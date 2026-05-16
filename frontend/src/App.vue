<script setup lang="ts">
import { computed, onMounted } from "vue";

import ChatComposer from "./components/ChatComposer.vue";
import HistoryPanel from "./components/HistoryPanel.vue";
import MessageList from "./components/MessageList.vue";
import StatusBanner from "./components/StatusBanner.vue";
import { useChatSession } from "./composables/useChatSession";

const {
  errorMessage,
  hasMessages,
  historyItems,
  infoMessage,
  isHistoryLoading,
  isLoading,
  isRestoring,
  messages,
  openThread,
  removeThread,
  removeTurn,
  resetSession,
  restoreSession,
  sendMessage,
  threadId,
  threadTitle,
} = useChatSession();

const activeConversationLabel = computed(() =>
  threadTitle.value.trim() || "未命名会话",
);

const consoleTitle = computed(() =>
  hasMessages.value ? activeConversationLabel.value : "想了解温商院的什么？",
);

const consoleSubtitle = computed(() =>
  hasMessages.value
    ? "会话会自动保存。"
    : "围绕学院、专业、课程、培养方案或就业方向直接提问。",
);

const sessionBadge = computed(() => (threadId.value ? "已保存" : "新会话"));

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

        <StatusBanner
          v-if="infoMessage"
          tone="info"
          :message="infoMessage"
        />
        <StatusBanner
          v-if="errorMessage"
          tone="error"
          :message="errorMessage"
        />

        <div class="console-body">
          <section
            class="conversation-shell"
            :class="{ 'conversation-shell--empty': !hasMessages && !isRestoring }"
          >
            <MessageList
              :messages="messages"
              :is-restoring="isRestoring"
              :disabled="isLoading || isRestoring"
              @remove-turn="removeTurn"
            />
          </section>

          <div class="composer-dock">
            <ChatComposer
              :disabled="isRestoring"
              :loading="isLoading"
              @submit="sendMessage"
            />
          </div>
        </div>
      </section>
    </main>
  </div>
</template>
