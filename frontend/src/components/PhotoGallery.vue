<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import type { Photo } from "../api";
import { formatDate } from "../api";

const props = defineProps<{ photos: Photo[] }>();
const selectedIndex = ref<number | null>(null);

const selected = computed(() =>
  selectedIndex.value !== null ? (props.photos[selectedIndex.value] ?? null) : null,
);

function open(index: number) {
  selectedIndex.value = index;
}

function close() {
  selectedIndex.value = null;
}

function step(delta: number) {
  if (selectedIndex.value === null || !props.photos.length) return;
  selectedIndex.value = (selectedIndex.value + delta + props.photos.length) % props.photos.length;
}

function onKey(event: KeyboardEvent) {
  if (event.key === "Escape") close();
  else if (event.key === "ArrowLeft") step(-1);
  else if (event.key === "ArrowRight") step(1);
}

watch(selected, (photo) => {
  if (photo) window.addEventListener("keydown", onKey);
  else window.removeEventListener("keydown", onKey);
});

onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
</script>

<template>
  <div v-if="photos.length" class="photos">
    <figure v-for="(photo, index) in photos" :key="photo.url">
      <button type="button" class="photo-open" @click="open(index)">
        <img :src="photo.thumbnail_url || photo.url" :alt="photo.caption" loading="lazy" decoding="async" />
      </button>
      <figcaption>
        <strong>
          <RouterLink v-if="photo.slug" :to="`/especies/${photo.slug}`" class="photo-link">
            {{ photo.species_short || "Avistamiento" }}
          </RouterLink>
          <template v-else>{{ photo.species_short || "Avistamiento" }}</template>
        </strong>
        <RouterLink :to="{ path: '/', query: { source: photo.source, date: photo.date } }" class="photo-date-link">
          {{ formatDate(photo.date) }}
        </RouterLink>
        · {{ photo.caption }}
      </figcaption>
    </figure>
  </div>
  <p v-else class="note">No hay fotos para este filtro.</p>
  <Teleport to="body">
    <div
      v-if="selected"
      class="lightbox"
      role="dialog"
      aria-modal="true"
      :aria-label="selected.caption"
      @click="close"
    >
      <button type="button" class="lightbox-btn lightbox-close" aria-label="Cerrar" @click.stop="close">×</button>
      <button
        v-if="photos.length > 1"
        type="button"
        class="lightbox-btn lightbox-prev"
        aria-label="Foto anterior"
        @click.stop="step(-1)"
      >
        ‹
      </button>
      <figure class="lightbox-frame" @click.stop>
        <img :src="selected.url" :alt="selected.caption" />
        <figcaption>
          <strong>
            <RouterLink v-if="selected.slug" :to="`/especies/${selected.slug}`" class="photo-link" @click="close">
              {{ selected.species_short || "Avistamiento" }}
            </RouterLink>
            <template v-else>{{ selected.species_short || "Avistamiento" }}</template>
          </strong>
          <RouterLink
            :to="{ path: '/', query: { source: selected.source, date: selected.date } }"
            class="photo-date-link"
            @click="close"
          >
            Ver {{ formatDate(selected.date) }} en el mapa
          </RouterLink>
          · {{ selected.caption }}
          <span v-if="photos.length > 1" class="lightbox-counter">
            ({{ (selectedIndex ?? 0) + 1 }} / {{ photos.length }})
          </span>
        </figcaption>
      </figure>
      <button
        v-if="photos.length > 1"
        type="button"
        class="lightbox-btn lightbox-next"
        aria-label="Foto siguiente"
        @click.stop="step(1)"
      >
        ›
      </button>
    </div>
  </Teleport>
</template>
