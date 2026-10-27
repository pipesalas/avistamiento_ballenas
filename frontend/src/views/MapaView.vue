<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import {
  formatDate,
  getBehavior,
  getEnvironment,
  getMeta,
  getPhotos,
  getShoreZones,
  getSightings,
  getSpeciesStats,
  getTracks,
  sightingsCsvUrl,
  type BehaviorRow,
  type FeatureCollection,
  type Meta,
  type Photo,
  type Species,
} from "../api";
import PhotoGallery from "../components/PhotoGallery.vue";
import SightingMap from "../components/SightingMap.vue";
import { rampFor } from "../scales";

const route = useRoute();
const router = useRouter();
const meta = ref<Meta | null>(null);
const stats = ref<Species[]>([]);
const error = ref("");
const source = ref<"boat" | "shore">(route.query.source === "shore" ? "shore" : "boat");
const species = ref(String(route.query.species ?? ""));
const behavior = ref(String(route.query.behavior ?? ""));
const variable = ref(String(route.query.variable ?? "temperature"));
const depth = ref(String(route.query.depth ?? "surface"));
const date = ref(String(route.query.date ?? ""));
const seaOn = ref(route.query.sea === "1");
const heatmap = ref(route.query.heatmap === "1");
const environment = ref<FeatureCollection | null>(null);
const tracks = ref<FeatureCollection | null>(null);
const zones = ref<FeatureCollection | null>(null);
const sightings = ref<FeatureCollection | null>(null);
const photos = ref<Photo[]>([]);
const behaviorRows = ref<BehaviorRow[]>([]);
const snappedNote = ref("");
const loading = ref(false);
const ready = ref(false);
const legendCollapsed = ref(typeof window !== "undefined" ? window.innerWidth <= 860 : false);
const photosCollapsed = ref(false);
let requestId = 0;

const empty: FeatureCollection = { type: "FeatureCollection", features: [] };

const activeSpecies = computed(() => stats.value.find((item) => item.slug === species.value) ?? null);

const activeDates = computed(() => {
  if (!meta.value) return [];
  return source.value === "shore" ? meta.value.sightings.shore_dates : meta.value.sightings.boat_dates;
});
const speciesChips = computed(() =>
  stats.value
    .map((item) => ({
      ...item,
      count: source.value === "shore" ? item.shore_events ?? 0 : item.boat_events ?? 0,
    }))
    .filter((item) => item.count > 0),
);
const scale = computed(() => rampFor(variable.value, meta.value).join(", "));
const unit = computed(() => meta.value?.environment.variables.find((item) => item.id === variable.value)?.unit ?? "");
const previousDay = computed(() => {
  const index = activeDates.value.indexOf(date.value);
  return index > 0 ? activeDates.value[index - 1] : "";
});
const showEmpty = computed(
  () => ready.value && !loading.value && !!date.value && (sightings.value?.features.length ?? 0) === 0,
);
const csvHref = computed(() => {
  const query: Record<string, string> = { source: source.value };
  if (date.value) query.date = date.value;
  if (species.value) query.species = species.value;
  if (behavior.value && species.value) query.behavior = behavior.value;
  return sightingsCsvUrl(query);
});

function datesFor(next: "boat" | "shore") {
  if (!meta.value) return [];
  return next === "shore" ? meta.value.sightings.shore_dates : meta.value.sightings.boat_dates;
}

function syncQuery() {
  const query: Record<string, string> = { source: source.value };
  if (date.value) query.date = date.value;
  if (species.value) query.species = species.value;
  if (behavior.value && species.value) query.behavior = behavior.value;
  if (heatmap.value) query.heatmap = "1";
  if (seaOn.value) {
    query.sea = "1";
    query.variable = variable.value;
    if (variable.value === "temperature") query.depth = depth.value;
  }
  router.replace({ query });
}

function setSource(next: "boat" | "shore") {
  behavior.value = "";
  source.value = next;
  if (!date.value) return;
  const list = datesFor(next);
  if (list.includes(date.value)) {
    snappedNote.value = "";
    return;
  }
  const earlier = [...list].reverse().find((day) => day <= date.value);
  date.value = earlier ?? "";
  snappedNote.value = date.value
    ? `Ese día no tiene registros de ${next === "boat" ? "barco" : "orilla"}. Mostrando el ${formatDate(date.value)}.`
    : "";
}

function toggleSpecies(slug: string) {
  behavior.value = "";
  species.value = species.value === slug ? "" : slug;
  snappedNote.value = "";
}

function toggleBehavior(value: string) {
  behavior.value = behavior.value === value ? "" : value;
}

function toggleSea() {
  seaOn.value = !seaOn.value;
  if (seaOn.value && !date.value && activeDates.value.length) {
    date.value = activeDates.value[activeDates.value.length - 1];
  }
}

function step(delta: number) {
  const list = activeDates.value;
  if (!list.length) return;
  snappedNote.value = "";
  if (!date.value) {
    if (delta < 0) date.value = list[list.length - 1];
    return;
  }
  const index = list.indexOf(date.value);
  const next = (index === -1 ? list.length - 1 : index) + delta;
  if (next < 0) return;
  date.value = next >= list.length ? "" : list[next];
}

function onCalendar(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  const list = activeDates.value;
  if (!value) {
    date.value = "";
    snappedNote.value = "";
    return;
  }
  if (list.includes(value)) {
    date.value = value;
    snappedNote.value = "";
    return;
  }
  const earlier = [...list].reverse().find((day) => day <= value);
  const later = list.find((day) => day >= value);
  date.value = earlier ?? later ?? "";
  snappedNote.value = date.value
    ? `Ese día no tiene registros. Mostrando el ${formatDate(date.value)}.`
    : "Ese día no tiene registros.";
}

async function load() {
  if (!meta.value) return;
  const id = ++requestId;
  loading.value = true;
  error.value = "";
  const sightingQuery: Record<string, string> = { source: source.value };
  if (date.value) sightingQuery.date = date.value;
  if (species.value) sightingQuery.species = species.value;
  if (behavior.value && species.value) sightingQuery.behavior = behavior.value;
  const zoneQuery: Record<string, string> = {};
  if (date.value) zoneQuery.date = date.value;
  if (species.value) zoneQuery.species = species.value;
  const photoQuery: Record<string, string> = { source: source.value };
  if (date.value) photoQuery.date = date.value;
  if (species.value) photoQuery.species = species.value;
  try {
    const [sightingValue, zoneValue, trackValue, behaviorValue, environmentValue, photoValue] = await Promise.all([
      getSightings(sightingQuery),
      source.value === "shore" ? getShoreZones(zoneQuery) : Promise.resolve(empty),
      source.value === "boat" ? getTracks(date.value || undefined) : Promise.resolve(empty),
      species.value ? getBehavior(species.value, source.value) : Promise.resolve([] as BehaviorRow[]),
      seaOn.value && date.value
        ? getEnvironment(variable.value, date.value, variable.value === "temperature" ? depth.value : "mean")
        : Promise.resolve(null),
      date.value ? getPhotos(photoQuery) : Promise.resolve([] as Photo[]),
    ]);
    if (id !== requestId) return;
    sightings.value = sightingValue;
    zones.value = zoneValue;
    tracks.value = trackValue;
    behaviorRows.value = behaviorValue;
    environment.value = environmentValue;
    photos.value = photoValue;
  } catch (cause) {
    if (id !== requestId) return;
    error.value = cause instanceof Error ? cause.message : "No se pudo cargar el mapa";
  } finally {
    if (id === requestId) loading.value = false;
  }
}

onMounted(async () => {
  try {
    const [metaValue, statsValue] = await Promise.all([getMeta(), getSpeciesStats("both")]);
    meta.value = metaValue;
    stats.value = statsValue;
    if (seaOn.value && !date.value && activeDates.value.length) {
      date.value = activeDates.value[activeDates.value.length - 1];
    }
    await load();
    ready.value = true;
    syncQuery();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "No se pudo cargar el mapa";
  }
});

watch([source, species, behavior, variable, depth, date, seaOn], () => {
  if (!ready.value) return;
  syncQuery();
  load();
});

// The heatmap only changes rendering; it uses the sightings already loaded.
watch(heatmap, () => {
  if (ready.value) syncQuery();
});
</script>

<template>
  <div class="map-page">
    <div class="map-stage">
      <div class="map-toolbar">
        <div class="toolbar-row">
          <div class="tabs">
            <button type="button" :class="{ active: source === 'boat' }" @click="setSource('boat')">
              Barco · {{ meta?.sightings.boat_count ?? "…" }}
            </button>
            <button type="button" :class="{ active: source === 'shore' }" @click="setSource('shore')">
              Orilla · {{ meta?.sightings.shore_count ?? "…" }}
            </button>
          </div>
          <div class="timebar">
            <button type="button" aria-label="Día anterior" :disabled="!!date && date === activeDates[0]" @click="step(-1)">‹</button>
            <select
              v-model="date"
              class="time-select"
              aria-label="Seleccionar fecha con registros"
              @change="snappedNote = ''"
            >
              <option value="">Todas las salidas ({{ activeDates.length }})</option>
              <option v-for="d in [...activeDates].reverse()" :key="d" :value="d">
                {{ formatDate(d) }}
              </option>
            </select>
            <button type="button" aria-label="Día siguiente" :disabled="!date" @click="step(1)">›</button>
            <input
              type="date"
              aria-label="Elegir fecha en calendario"
              :value="date"
              :min="meta?.sightings.start"
              :max="meta?.sightings.end"
              @change="onCalendar"
            />
          </div>
          <div class="toolbar-actions">
            <button type="button" class="layer-toggle" :class="{ active: seaOn }" @click="toggleSea">Capa del mar</button>
            <button
              type="button"
              class="layer-toggle"
              :class="{ active: heatmap }"
              @click="heatmap = !heatmap"
            >
              Mapa de calor
            </button>
            <a :href="csvHref" download class="layer-toggle link-chip">Descargar CSV</a>
            <span v-if="loading" class="note toolbar-status">Cargando…</span>
            <span v-else-if="sightings" class="note toolbar-status">
              {{ sightings.features.length }} avistamientos
              <template v-if="source === 'shore' && zones"> · {{ zones.features.length }} subzonas</template>
            </span>
          </div>
        </div>
        <div v-if="seaOn" class="toolbar-row sea-controls">
          <label class="field inline">
            <span>Variable</span>
            <select v-model="variable">
              <option v-for="item in meta?.environment.variables" :key="item.id" :value="item.id">{{ item.label }}</option>
            </select>
          </label>
          <label v-if="variable === 'temperature'" class="field inline">
            <span>Profundidad</span>
            <select v-model="depth">
              <option value="surface">Superficie ({{ meta?.environment.surface_depth }} m)</option>
              <option value="mean">Promedio de la columna</option>
              <option v-for="item in meta?.environment.depths" :key="item" :value="String(item)">{{ item }} m</option>
            </select>
          </label>
        </div>
        <div v-if="species && behaviorRows.length" class="toolbar-row chips">
          <button type="button" :class="{ active: !behavior }" @click="behavior = ''">Todos</button>
          <button
            v-for="item in behaviorRows"
            :key="item.behavior"
            type="button"
            :class="{ active: behavior === item.behavior }"
            @click="toggleBehavior(item.behavior)"
          >
            {{ item.behavior }} · {{ item.events }}
          </button>
        </div>
        <p v-if="error" class="error">{{ error }}</p>
        <p v-else-if="snappedNote" class="note">{{ snappedNote }}</p>
      </div>
      <div class="map-canvas">
      <SightingMap
        v-if="ready"
        :environment="seaOn && date ? environment : null"
        :tracks="source === 'boat' ? tracks : null"
        :zones="source === 'shore' ? zones : null"
        :sightings="sightings"
        :fit="!!date"
        :heatmap="heatmap"
      />
      <p v-if="showEmpty" class="map-empty">
        Este día no hay avistamientos con este filtro.
        <button v-if="previousDay" type="button" @click="date = previousDay">Ver {{ formatDate(previousDay) }}</button>
        <button v-if="species" type="button" @click="toggleSpecies(species)">Quitar especie</button>
      </p>
      <aside class="legend" :class="{ raised: !!date, collapsed: legendCollapsed }">
        <div class="legend-header">
          <div class="legend-title">
            <strong>Leyenda y filtro</strong>
            <span v-if="legendCollapsed && activeSpecies" class="legend-active-pill">
              <i :style="{ background: activeSpecies.color }"></i>
              {{ activeSpecies.short }}
            </span>
          </div>
          <button
            type="button"
            class="legend-toggle"
            :aria-expanded="!legendCollapsed"
            @click="legendCollapsed = !legendCollapsed"
          >
            {{ legendCollapsed ? "Mostrar" : "Ocultar" }}
          </button>
        </div>
        <div v-show="!legendCollapsed" class="legend-body">
          <template v-if="seaOn && date && environment?.properties">
            <h3>{{ variable === "temperature" ? "Temperatura" : variable === "chlorophyll" ? "Clorofila" : "Fitoplancton" }}</h3>
            <div class="scale" :style="{ background: `linear-gradient(90deg, ${scale})` }"></div>
            <div>{{ environment.properties.min }} – {{ environment.properties.max }} {{ unit }}</div>
          </template>
          <h3>Especies</h3>
          <button
            v-for="item in speciesChips"
            :key="item.slug"
            type="button"
            class="swatch"
            :class="{ active: species === item.slug }"
            @click="toggleSpecies(item.slug)"
          >
            <i :style="{ background: item.color }"></i>
            <span>{{ item.short }} · {{ item.count }}</span>
          </button>
        </div>
      </aside>
      </div>
      <section v-if="date" class="photo-strip" :class="{ empty: ready && !photos.length, collapsed: photosCollapsed }">
        <div class="photo-strip-header">
          <h2>Fotos<template v-if="ready && photos.length"> ({{ photos.length }})</template></h2>
          <button
            v-if="ready && photos.length"
            type="button"
            class="legend-toggle"
            :aria-expanded="!photosCollapsed"
            @click="photosCollapsed = !photosCollapsed"
          >
            {{ photosCollapsed ? "Mostrar" : "Ocultar" }}
          </button>
        </div>
        <template v-if="!photosCollapsed">
          <p v-if="!ready" class="note">Cargando…</p>
          <PhotoGallery v-else :photos="photos" />
        </template>
      </section>
    </div>
  </div>
</template>
