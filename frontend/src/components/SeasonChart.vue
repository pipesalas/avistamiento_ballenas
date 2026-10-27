<script setup lang="ts">
import { HeatmapChart } from "echarts/charts";
import { GridComponent, TooltipComponent, VisualMapComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { SeasonRow } from "../api";

echarts.use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

const props = defineProps<{ rows: SeasonRow[] }>();
const months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"];
const el = ref<HTMLDivElement | null>(null);
let chart: echarts.ECharts | null = null;
let observer: ResizeObserver | null = null;

function render() {
  if (!chart) return;
  const compact = (el.value?.clientWidth ?? 800) < 520;
  const years = [...new Set(props.rows.map((row) => row.year))].sort();
  const cells = new Map<string, number>();
  for (const row of props.rows) {
    const key = `${row.year}-${row.month}`;
    cells.set(key, (cells.get(key) ?? 0) + row.individuals);
  }
  const data = years.flatMap((year, yearIndex) =>
    months.map((_, monthIndex) => [monthIndex, yearIndex, cells.get(`${year}-${monthIndex + 1}`) ?? 0]),
  );
  chart.setOption({
    tooltip: {
      confine: true,
      formatter: (params: unknown) => {
        const point = params as { data: number[] };
        const [month, yearIndex, value] = point.data;
        return `${months[month]} ${years[yearIndex]}: ${value} ejemplares`;
      },
    },
    grid: { left: compact ? 42 : 56, right: compact ? 8 : 16, bottom: compact ? 40 : 32, top: 16 },
    xAxis: {
      type: "category",
      data: months,
      name: compact ? "" : "Mes",
      axisLabel: { interval: 0, rotate: compact ? 40 : 0, fontSize: compact ? 10 : 12 },
    },
    yAxis: { type: "category", data: years.map(String), name: compact ? "" : "Año" },
    visualMap: {
      min: 0,
      max: Math.max(1, ...data.map((cell) => cell[2])),
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      show: false,
      inRange: { color: ["#f3f3f3", "#06C167"] },
    },
    series: [{
      type: "heatmap",
      data,
      label: {
        show: true,
        fontSize: compact ? 10 : 12,
        formatter: (params: unknown) => {
          const point = params as { data: number[] };
          return point.data[2] ? String(point.data[2]) : "";
        },
      },
    }],
  });
}

onMounted(() => {
  if (!el.value) return;
  chart = echarts.init(el.value);
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
  <div ref="el" class="chart short"></div>
</template>
