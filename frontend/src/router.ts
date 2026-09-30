import { createRouter, createWebHistory } from "vue-router";
import ResumenView from "./views/ResumenView.vue";
import MapaView from "./views/MapaView.vue";
import EspecieView from "./views/EspecieView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "mapa", component: MapaView },
    { path: "/datos", name: "datos", component: ResumenView },
    { path: "/mapa", redirect: (to) => ({ path: "/", query: to.query }) },
    { path: "/especies/:slug", name: "especie", component: EspecieView, props: true },
  ],
});

export default router;
