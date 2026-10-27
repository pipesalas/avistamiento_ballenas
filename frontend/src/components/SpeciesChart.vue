<script setup lang="ts">
import { BarChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import type { Species } from "../api";

echarts.use([BarChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

const props = defineProps<{ rows: Species[] }>();
const el = ref<HTMLDivElement | null>(null);
const router = useRouter();
let chart: echarts.ECharts | null = null;
let observer: ResizeObserver | null = null;

function render() {
  if (!chart) return;
  const compact = (el.value?.clientWidth ?? 800) < 520;
  const names = props.rows.map((row) => row.short);
  chart.setOption({
    tooltip: { trigger: "axis", confine: true },
    legend: { data: ["Barco", "Orilla"] },
    grid: { left: compact ? 40 : 48, right: compact ? 10 : 16, bottom: compact ? 92 : 80, top: 32 },
    xAxis: {
      type: "category",
      data: names,
      axisLabel: { rotate: compact ? 42 : 35, interval: 0, fontSize: compact ? 11 : 12 },
    },
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
