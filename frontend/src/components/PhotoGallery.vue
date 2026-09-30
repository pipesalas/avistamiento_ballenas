<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";
import type { Photo } from "../api";
import { formatDate } from "../api";

defineProps<{ photos: Photo[] }>();
const selected = ref<Photo | null>(null);

function open(photo: Photo) {
  selected.value = photo;
}

function close() {
  selected.value = null;
}

function onKey(event: KeyboardEvent) {
  if (event.key === "Escape") close();
}

watch(selected, (photo) => {
  if (photo) window.addEventListener("keydown", onKey);
  else window.removeEventListener("keydown", onKey);
});

onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
</script>

<template>
  <div v-if="photos.length" class="photos">
    <figure v-for="photo in photos" :key="photo.url">
      <button type="button" class="photo-open" @click="open(photo)">
        <img :src="photo.url" :alt="photo.caption" />
      </button>
      <figcaption>
        <strong>
          {{ photo.species_short || "Avistamiento" }}
        </strong>
        {{ formatDate(photo.date) }} · {{ photo.caption }}
      </figcaption>
    </figure>
  </div>
  <p v-else class="note">No hay fotos para este filtro.</p>
  <Teleport to="body">
    <div v-if="selected" class="lightbox" @click="close">
      <figure class="lightbox-frame" @click.stop>
        <img :src="selected.url" :alt="selected.caption" />
        <figcaption>
          <strong>{{ selected.species_short || "Avistamiento" }}</strong>
          {{ formatDate(selected.date) }} · {{ selected.caption }}
        </figcaption>
      </figure>
    </div>
  </Teleport>
</template>
