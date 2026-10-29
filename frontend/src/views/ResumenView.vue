<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import {
  getBehavior,
  getMeta,
  getSeasonality,
  getSpeciesStats,
  getTimeline,
  type BehaviorRow,
  type Meta,
  type SeasonRow,
  type Species,
  type TimelineRow,
} from "../api";
import BehaviorChart from "../components/BehaviorChart.vue";
import SeasonChart from "../components/SeasonChart.vue";
import SpeciesChart from "../components/SpeciesChart.vue";
import TimelineChart from "../components/TimelineChart.vue";

const meta = ref<Meta | null>(null);
const species = ref<Species[]>([]);
const season = ref<SeasonRow[]>([]);
const timeline = ref<TimelineRow[]>([]);
const behavior = ref<BehaviorRow[]>([]);
const error = ref("");
const loading = ref(true);

const totalSightings = computed(() => {
  if (!meta.value) return 0;
  return meta.value.sightings.boat_count + meta.value.sightings.shore_count;
});

const totalIndividuals = computed(() => {
  if (meta.value?.sightings.boat_individuals !== undefined && meta.value?.sightings.shore_individuals !== undefined) {
    return meta.value.sightings.boat_individuals + meta.value.sightings.shore_individuals;
  }
  return species.value.reduce((acc, row) => acc + (row.boat_individuals ?? 0) + (row.shore_individuals ?? 0), 0);
});

onMounted(async () => {
  try {
    const [metaValue, speciesValue, seasonValue, timelineValue, behaviorValue] = await Promise.all([
      getMeta(),
      getSpeciesStats("both"),
      getSeasonality("both"),
      getTimeline("both", "day"),
      getBehavior(),
    ]);
    meta.value = metaValue;
    species.value = speciesValue;
    season.value = seasonValue;
    timeline.value = timelineValue;
    behavior.value = behaviorValue;
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "No se pudo cargar el resumen";
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <main class="page">
    <p v-if="error" class="error">{{ error }}</p>
    <div v-else-if="loading" class="skeleton-stack" aria-busy="true">
      <div class="loading-banner" role="status" aria-live="polite">
        <span class="spinner" aria-hidden="true"></span>
        <span>Cargando datos del monitoreo…</span>
      </div>
      <div class="skeleton hero-skeleton"></div>
      <div class="kpi-grid">
        <div v-for="n in 4" :key="n" class="skeleton kpi-skeleton"></div>
      </div>
      <div class="skeleton panel-skeleton"></div>
    </div>
    <template v-else-if="meta">
      <section class="hero">
        <p class="eyebrow">Vuelve al Océano</p>
        <h1>{{ meta.project.title }}</h1>
        <p class="lead">{{ meta.project.caption }}</p>
        <p class="intro">
          El objetivo de este proyecto es conocer qué especies de mamíferos marinos transitan por el área de estudio
          (Huiro y Chaihuín, comuna de Corral, Los Ríos, Chile) y qué comportamientos tienen en la zona. ¿Se alimentan,
          descansan o se reproducen? ¿Cuántos individuos son, regresan todos los años, cuándo están y cuánto tiempo
          permanecen? El monitoreo, desde 2022, combina navegaciones del equipo y reportes de vecinos y vecinas por
          WhatsApp.
        </p>
        <div class="hero-actions">
          <RouterLink class="cta" to="/">Ver el mapa</RouterLink>
        </div>
      </section>
      <section class="kpi-grid" aria-label="Indicadores generales">
        <div class="kpi-card">
          <span class="kpi-label">Avistamientos totales</span>
          <strong class="kpi-value">{{ totalSightings.toLocaleString("es-CL") }}</strong>
          <span class="kpi-sub">{{ meta.sightings.boat_count }} barco · {{ meta.sightings.shore_count }} orilla</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Ejemplares contabilizados</span>
          <strong class="kpi-value">{{ totalIndividuals.toLocaleString("es-CL") }}</strong>
          <span class="kpi-sub">Suma acumulada del monitoreo</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Especies registradas</span>
          <strong class="kpi-value">{{ species.length }}</strong>
          <span class="kpi-sub">Cetáceos, pinnípedos y mustélidos</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Días con registro</span>
          <strong class="kpi-value">{{ meta.sightings.dates.length }}</strong>
          <span class="kpi-sub">{{ meta.sightings.boat_dates.length }} salidas en barco</span>
        </div>
      </section>
      <section class="panel">
        <h2>Ejemplares por especie</h2>
        <p class="note">Barco y orilla por separado. Un clic abre la ficha de la especie.</p>
        <SpeciesChart :rows="species" />
      </section>
      <section class="panel">
        <h2>Línea de tiempo por salida</h2>
        <p class="note">Usa la barra inferior o la rueda del mouse para hacer zoom. Un clic en una fecha la abre en el mapa.</p>
        <TimelineChart :rows="timeline" />
      </section>
      <section class="panel">
        <h2>Estacionalidad</h2>
        <p class="note">Ejemplares por mes y año, sumando barco y orilla.</p>
        <SeasonChart :rows="season" />
      </section>
      <section class="panel">
        <h2>Comportamiento observado</h2>
        <p class="note">Registros de barco y orilla. Un clic abre el mapa filtrado por esa especie y comportamiento.</p>
        <BehaviorChart :rows="behavior" />
      </section>
    </template>
  </main>
</template>
