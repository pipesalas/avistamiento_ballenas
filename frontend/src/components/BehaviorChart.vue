<script setup lang="ts">
import { BarChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { BehaviorRow } from "../api";

echarts.use([BarChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

const props = defineProps<{ rows: BehaviorRow[] }>();
const el = ref<HTMLDivElement | null>(null);
const router = useRouter();
let chart: echarts.ECharts | null = null;
let observer: ResizeObserver | null = null;

function render() {
  if (!chart) return;
  const compact = (el.value?.clientWidth ?? 800) < 520;
  const species = [...new Set(props.rows.map((row) => row.species_short))];
  const behaviors = [...new Set(props.rows.map((row) => row.behavior))];
  chart.setOption({
    color: ["#000000", "#06C167", "#545454", "#B7B7B7", "#0B3D24", "#8EE0B0"],
    tooltip: { trigger: "axis", confine: true },
    legend: { type: "scroll" },
    grid: { left: compact ? 40 : 48, right: compact ? 10 : 16, bottom: compact ? 92 : 80, top: 40 },
    xAxis: {
      type: "category",
      data: species,
      axisLabel: { rotate: compact ? 42 : 35, interval: 0, fontSize: compact ? 11 : 12 },
    },
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
  observer = new ResizeObserver(() => {
    render();
    chart?.resize();
  });
  observer.observe(el.value);
});

watch(() => props.rows, render);
onBeforeUnmount(() => {
  observer?.disconnect();
  chart?.dispose();
});
</script>

<template>
  <div ref="el" class="chart"></div>
</template>
