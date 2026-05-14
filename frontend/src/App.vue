<script setup lang="ts">
import { onMounted } from "vue";

import ChatComposer from "./components/ChatComposer.vue";
import ChatHeader from "./components/ChatHeader.vue";
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

onMounted(() => {
  void restoreSession();
});
</script>

<template>
  <div class="page-shell">
    <div class="page-shell__ornament page-shell__ornament--top" />
    <div class="page-shell__ornament page-shell__ornament--bottom" />

    <main class="workspace">
      <HistoryPanel
        :active-thread-id="threadId"
        :items="historyItems"
        :disabled="isLoading || isRestoring"
        :loading="isHistoryLoading"
        @select="openThread"
        @remove="removeThread"
      />

      <section class="console">
        <ChatHeader
          :thread-id="threadId"
          :has-messages="hasMessages"
          :disabled="isLoading || isRestoring"
          @reset="resetSession"
        />

        <StatusBanner v-if="threadId" tone="info" :message="`Current session: ${threadTitle}`" />
        <StatusBanner v-if="infoMessage" tone="info" :message="infoMessage" />
        <StatusBanner v-if="errorMessage" tone="error" :message="errorMessage" />

        <MessageList
          :messages="messages"
          :is-restoring="isRestoring"
          :disabled="isLoading || isRestoring"
          @remove-turn="removeTurn"
        />

        <ChatComposer
          :disabled="isRestoring"
          :loading="isLoading"
          @submit="sendMessage"
        />
      </section>
    </main>
  </div>
</template>
