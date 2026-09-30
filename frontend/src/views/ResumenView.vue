<script setup lang="ts">
import { onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { getBehavior, getMeta, getSeasonality, getSpeciesStats, type BehaviorRow, type Meta, type SeasonRow, type Species } from "../api";
import BehaviorChart from "../components/BehaviorChart.vue";
import SeasonChart from "../components/SeasonChart.vue";
import SpeciesChart from "../components/SpeciesChart.vue";

const meta = ref<Meta | null>(null);
const species = ref<Species[]>([]);
const season = ref<SeasonRow[]>([]);
const behavior = ref<BehaviorRow[]>([]);
const error = ref("");

onMounted(async () => {
  try {
    const [metaValue, speciesValue, seasonValue, behaviorValue] = await Promise.all([
      getMeta(),
      getSpeciesStats("both"),
      getSeasonality("both"),
      getBehavior(),
    ]);
    meta.value = metaValue;
    species.value = speciesValue;
    season.value = seasonValue;
    behavior.value = behaviorValue;
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "No se pudo cargar el resumen";
  }
});
</script>

<template>
  <main class="page">
    <p v-if="error" class="error">{{ error }}</p>
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
      <RouterLink class="cta" to="/">Ver el mapa</RouterLink>
      </section>
      <section class="panel">
        <h2>Ejemplares por especie</h2>
        <p class="note">Barco y orilla por separado. Un clic abre la ficha de la especie.</p>
        <SpeciesChart :rows="species" />
      </section>
      <section class="panel">
        <h2>Estacionalidad</h2>
        <p class="note">Ejemplares por mes y año, sumando barco y orilla.</p>
        <SeasonChart :rows="season" />
      </section>
      <section class="panel">
        <h2>Comportamiento en navegación</h2>
        <p class="note">Un clic abre el mapa filtrado por esa especie y ese comportamiento.</p>
        <BehaviorChart :rows="behavior" />
      </section>
    </template>
  </main>
</template>
