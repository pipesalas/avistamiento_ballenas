<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { RouterLink, useRoute } from "vue-router";
import {
  formatDate,
  getBehavior,
  getMeta,
  getPhotos,
  getSeasonality,
  getSightings,
  type BehaviorRow,
  type FeatureCollection,
  type Meta,
  type Photo,
  type SeasonRow,
} from "../api";
import BehaviorChart from "../components/BehaviorChart.vue";
import PhotoGallery from "../components/PhotoGallery.vue";
import SeasonChart from "../components/SeasonChart.vue";
import SightingMap from "../components/SightingMap.vue";

const props = defineProps<{ slug: string }>();
const route = useRoute();
const meta = ref<Meta | null>(null);
const season = ref<SeasonRow[]>([]);
const behavior = ref<BehaviorRow[]>([]);
const sightings = ref<FeatureCollection | null>(null);
const photos = ref<Photo[]>([]);
const error = ref("");
const loaded = ref(false);

const species = computed(() => meta.value?.species.find((item) => item.slug === props.slug) ?? null);

async function load() {
  error.value = "";
  loaded.value = false;
  try {
    meta.value = await getMeta();
    if (!species.value) {
      error.value = "No encontramos esa especie.";
      return;
    }
    const [seasonValue, behaviorValue, sightingValue, photoValue] = await Promise.all([
      getSeasonality("both", props.slug),
      getBehavior(props.slug),
      getSightings({ species: props.slug }),
      getPhotos({ species: props.slug }),
    ]);
    season.value = seasonValue;
    behavior.value = behaviorValue;
    sightings.value = sightingValue;
    photos.value = photoValue;
    loaded.value = true;
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "No se pudo cargar la ficha";
  }
}

onMounted(load);
watch(() => route.params.slug, load);
</script>

<template>
  <main class="page">
    <p v-if="error" class="error">{{ error }}</p>
    <template v-else-if="species">
      <h1>
        <i class="swatch-dot" :style="{ background: species.color }"></i>
        {{ species.short }}
      </h1>
      <p class="lead">{{ species.name }}</p>
      <RouterLink class="cta" :to="{ path: '/', query: { species: species.slug } }">Ver en el mapa</RouterLink>
      <section class="panel">
        <h2>Fotos</h2>
        <p v-if="!loaded" class="note">Cargando…</p>
        <PhotoGallery v-else :photos="photos" />
      </section>
      <section class="panel">
        <h2>Dónde se vio</h2>
        <div class="species-map">
          <SightingMap v-if="loaded" :environment="null" :tracks="null" :zones="null" :sightings="sightings" />
        </div>
      </section>
      <section class="panel">
        <h2>Avistamientos</h2>
        <table class="table">
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Fuente</th>
              <th>Ejemplares</th>
              <th>Comportamiento</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(feature, index) in sightings?.features" :key="index">
              <td>{{ formatDate(String(feature.properties?.date)) }}</td>
              <td>{{ feature.properties?.source === "boat" ? "Barco" : "Orilla" }}</td>
              <td>{{ feature.properties?.individuals }}</td>
              <td>{{ feature.properties?.behavior || "—" }}</td>
            </tr>
          </tbody>
        </table>
      </section>
      <section class="panel">
        <h2>Estacionalidad</h2>
        <SeasonChart :rows="season" />
      </section>
      <section v-if="behavior.length" class="panel">
        <h2>Comportamiento</h2>
        <BehaviorChart :rows="behavior" />
      </section>
    </template>
  </main>
</template>

<style scoped>
.swatch-dot {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  margin-right: 8px;
}
.species-map {
  position: relative;
  height: 420px;
}
</style>
