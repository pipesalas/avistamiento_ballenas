<script setup lang="ts">
import { RouterLink, RouterView, useRoute } from "vue-router";

const route = useRoute();
</script>

<template>
  <header class="app-header">
    <RouterLink class="brand" to="/">
      <img :src="'/media/logo'" alt="Vuelve al Océano" />
      <div>
        <strong>Monitoreo de mamíferos marinos de la costa de Corral</strong>
        <span>Mamíferos marinos</span>
      </div>
    </RouterLink>
    <nav>
      <RouterLink to="/" custom v-slot="{ href, navigate }">
        <a :href="href" @click="navigate" :class="{ 'router-link-active': route.path === '/' }">Mapa</a>
      </RouterLink>
      <RouterLink to="/datos">Datos</RouterLink>
    </nav>
  </header>
  <RouterView />
</template>
