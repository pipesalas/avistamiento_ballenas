<script setup lang="ts">
import * as echarts from "echarts";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { Species } from "../api";

const props = defineProps<{ rows: Species[] }>();
const el = ref<HTMLDivElement | null>(null);
const router = useRouter();
let chart: echarts.ECharts | null = null;

function render() {
  if (!chart) return;
  const names = props.rows.map((row) => row.short);
  chart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["Barco", "Orilla"] },
    grid: { left: 48, right: 16, bottom: 80, top: 32 },
    xAxis: { type: "category", data: names, axisLabel: { rotate: 35, interval: 0 } },
    yAxis: { type: "value", name: "Ejemplares" },
    series: [
      {
        name: "Barco",
        type: "bar",
        data: props.rows.map((row) => row.boat_individuals ?? 0),
        itemStyle: { color: "#000000", borderRadius: [6, 6, 0, 0] },
      },
      {
        name: "Orilla",
        type: "bar",
        data: props.rows.map((row) => row.shore_individuals ?? 0),
        itemStyle: { color: "#06C167", borderRadius: [6, 6, 0, 0] },
      },
    ],
  });
}

onMounted(() => {
  if (!el.value) return;
  chart = echarts.init(el.value);
  chart.on("click", (params) => {
    const row = props.rows.find((item) => item.short === params.name);
    if (row) router.push(`/especies/${row.slug}`);
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
