<script setup lang="ts">
import * as echarts from "echarts";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { BehaviorRow } from "../api";

const props = defineProps<{ rows: BehaviorRow[] }>();
const el = ref<HTMLDivElement | null>(null);
const router = useRouter();
let chart: echarts.ECharts | null = null;

function render() {
  if (!chart) return;
  const species = [...new Set(props.rows.map((row) => row.species_short))];
  const behaviors = [...new Set(props.rows.map((row) => row.behavior))];
  chart.setOption({
    color: ["#000000", "#06C167", "#545454", "#B7B7B7", "#0B3D24", "#8EE0B0"],
    tooltip: { trigger: "axis" },
    legend: { type: "scroll" },
    grid: { left: 48, right: 16, bottom: 80, top: 40 },
    xAxis: { type: "category", data: species, axisLabel: { rotate: 35, interval: 0 } },
    yAxis: { type: "value", name: "Registros" },
    series: behaviors.map((behavior) => ({
      name: behavior,
      type: "bar",
      stack: "comportamiento",
      data: species.map((name) => {
        const row = props.rows.find((item) => item.species_short === name && item.behavior === behavior);
        return row?.events ?? 0;
      }),
    })),
  });
}

onMounted(() => {
  if (!el.value) return;
  chart = echarts.init(el.value);
  chart.on("click", (params) => {
    const row = props.rows.find(
      (item) => item.species_short === params.name && item.behavior === params.seriesName,
    );
    if (!row) return;
    router.push({ path: "/", query: { source: "boat", species: row.slug, behavior: row.behavior } });
  });
  render();
  window.addEventListener("resize", resize);
});

function resize() {
  chart?.resize();
}

watch(() => props.rows, render);
onBeforeUnmount(() => {
  window.removeEventListener("resize", resize);
  chart?.dispose();
});
</script>

<template>
  <div ref="el" class="chart"></div>
</template>
