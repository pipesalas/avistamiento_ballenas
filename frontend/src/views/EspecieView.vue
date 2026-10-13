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
  sightingsCsvUrl,
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
const mapRef = ref<InstanceType<typeof SightingMap> | null>(null);
let requestId = 0;

const species = computed(() => meta.value?.species.find((item) => item.slug === props.slug) ?? null);

const sortedSightings = computed(() => {
  const features = sightings.value?.features ?? [];
  return [...features]
    .filter((f) => tableFilter.value === "all" || f.properties?.source === tableFilter.value)
    .sort((a, b) => String(b.properties?.date ?? "").localeCompare(String(a.properties?.date ?? "")));
});

function focusFeature(feature: { geometry: { coordinates: unknown } }) {
  const coords = feature.geometry.coordinates;
  if (Array.isArray(coords) && typeof coords[0] === "number" && typeof coords[1] === "number") {
    mapRef.value?.flyToPoint(coords[0], coords[1]);
  }
}

function formatOcean(propsObj?: Record<string, unknown> | null): string {
  if (!propsObj) return "—";
  const parts: string[] = [];
  if (propsObj.sst_c !== null && propsObj.sst_c !== undefined) parts.push(`${propsObj.sst_c} °C`);
  if (propsObj.chlorophyll !== null && propsObj.chlorophyll !== undefined) parts.push(`${propsObj.chlorophyll} mg/m³`);
  return parts.length ? parts.join(" · ") : "—";
}

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
      <div class="hero-actions">
        <RouterLink class="cta" :to="{ path: '/', query: { species: species.slug } }">Ver en el mapa</RouterLink>
        <a class="cta secondary" :href="sightingsCsvUrl({ species: species.slug })" download>Descargar CSV</a>
      </div>
      <section class="panel">
        <h2>Fotos</h2>
        <p v-if="!loaded" class="note">Cargando…</p>
        <PhotoGallery v-else :photos="photos" />
      </section>
      <section class="panel">
        <h2>Dónde se vio</h2>
        <p class="note">Puntos con borde blanco corresponden a salidas en barco; polígonos y puntos con borde negro a subzonas de orilla.</p>
        <div class="species-map">
          <SightingMap
            v-if="loaded"
            ref="mapRef"
            :environment="null"
            :tracks="null"
            :zones="zones"
            :sightings="sightings"
            :fit="true"
          />
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
        <p class="note">Haz clic en una fila para enfocar el punto arriba, o en la fecha para abrir ese día en el mapa principal.</p>
        <div class="table-scroll">
          <table class="table interactive-table">
            <thead>
              <tr>
                <th>Fecha</th>
                <th>Fuente</th>
                <th>Ejemplares</th>
                <th>Condición del mar</th>
                <th>Comportamiento / Notas</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(feature, index) in sortedSightings"
                :key="index"
                class="clickable-row"
                @click="focusFeature(feature)"
              >
                <td>
                  <RouterLink
                    :to="{
                      path: '/',
                      query: {
                        source: String(feature.properties?.source ?? 'boat'),
                        date: String(feature.properties?.date ?? ''),
                        species: species.slug,
                      },
                    }"
                    class="row-date-link"
                    @click.stop
                  >
                    {{ formatDate(String(feature.properties?.date)) }}
                  </RouterLink>
                </td>
                <td>
                  {{
                    feature.properties?.source === "boat"
                      ? "Barco"
                      : `Orilla (Subzona ${feature.properties?.subzone ?? "—"})`
                  }}
                </td>
                <td>{{ feature.properties?.individuals }}</td>
                <td>{{ formatOcean(feature.properties) }}</td>
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
