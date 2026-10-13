import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "mapa", component: () => import("./views/MapaView.vue") },
    { path: "/datos", name: "datos", component: () => import("./views/ResumenView.vue") },
    { path: "/mapa", redirect: (to) => ({ path: "/", query: to.query }) },
    { path: "/especies/:slug", name: "especie", component: () => import("./views/EspecieView.vue"), props: true },
  ],
});

export default router;
