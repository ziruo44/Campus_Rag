<script setup lang="ts">
defineProps<{
  message: string;
  tone: "error" | "info";
}>();
</script>

<template>
  <div class="status-banner" :class="`status-banner--${tone}`" role="status">
    <span class="status-banner__dot" />
    <p>{{ message }}</p>
  </div>
</template>
