<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import {
  formatDate,
  getBehavior,
  getMeta,
  getPhotos,
  getSeasonality,
  getShoreZones,
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
const meta = ref<Meta | null>(null);
const season = ref<SeasonRow[]>([]);
const behavior = ref<BehaviorRow[]>([]);
const sightings = ref<FeatureCollection | null>(null);
const zones = ref<FeatureCollection | null>(null);
const photos = ref<Photo[]>([]);
const error = ref("");
const loaded = ref(false);
const tableFilter = ref<"all" | "boat" | "shore">("all");
let requestId = 0;

const species = computed(() => meta.value?.species.find((item) => item.slug === props.slug) ?? null);

const sortedSightings = computed(() => {
  const features = sightings.value?.features ?? [];
  return [...features]
    .filter((f) => tableFilter.value === "all" || f.properties?.source === tableFilter.value)
    .sort((a, b) => String(b.properties?.date ?? "").localeCompare(String(a.properties?.date ?? "")));
});

async function load() {
  const id = ++requestId;
  error.value = "";
  loaded.value = false;
  try {
    const metaValue = await getMeta();
    if (id !== requestId) return;
    meta.value = metaValue;
    if (!species.value) {
      error.value = "No encontramos esa especie.";
      return;
    }
    const [seasonValue, behaviorValue, sightingValue, zoneValue, photoValue] = await Promise.all([
      getSeasonality("both", props.slug),
      getBehavior(props.slug),
      getSightings({ species: props.slug }),
      getShoreZones({ species: props.slug }),
      getPhotos({ species: props.slug }),
    ]);
    if (id !== requestId) return;
    season.value = seasonValue;
    behavior.value = behaviorValue;
    sightings.value = sightingValue;
    zones.value = zoneValue;
    photos.value = photoValue;
    loaded.value = true;
  } catch (cause) {
    if (id !== requestId) return;
    error.value = cause instanceof Error ? cause.message : "No se pudo cargar la ficha";
  }
}

onMounted(load);
watch(() => props.slug, load);
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
        <p class="note">Puntos con borde blanco corresponden a salidas en barco; polígonos y puntos con borde negro a subzonas de orilla.</p>
        <div class="species-map">
          <SightingMap v-if="loaded" :environment="null" :tracks="null" :zones="zones" :sightings="sightings" :fit="true" />
        </div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2>Avistamientos ({{ sortedSightings.length }})</h2>
          <div class="chips">
            <button type="button" :class="{ active: tableFilter === 'all' }" @click="tableFilter = 'all'">Todos</button>
            <button type="button" :class="{ active: tableFilter === 'boat' }" @click="tableFilter = 'boat'">Barco</button>
            <button type="button" :class="{ active: tableFilter === 'shore' }" @click="tableFilter = 'shore'">Orilla</button>
          </div>
        </div>
        <div class="table-scroll">
          <table class="table">
            <thead>
              <tr>
                <th>Fecha</th>
                <th>Fuente</th>
                <th>Ejemplares</th>
                <th>Comportamiento / Notas</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(feature, index) in sortedSightings" :key="index">
                <td>{{ formatDate(String(feature.properties?.date)) }}</td>
                <td>
                  {{
                    feature.properties?.source === "boat"
                      ? "Barco"
                      : `Orilla (Subzona ${feature.properties?.subzone ?? "—"})`
                  }}
                </td>
                <td>{{ feature.properties?.individuals }}</td>
                <td>{{ feature.properties?.behavior || feature.properties?.notes || "—" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
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
