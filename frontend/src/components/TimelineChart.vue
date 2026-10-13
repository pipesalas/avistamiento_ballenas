<script setup lang="ts">
import { BarChart } from "echarts/charts";
import { DataZoomComponent, GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { formatDate, type TimelineRow } from "../api";

echarts.use([BarChart, DataZoomComponent, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

const props = defineProps<{ rows: TimelineRow[] }>();
const el = ref<HTMLDivElement | null>(null);
const router = useRouter();
let chart: echarts.ECharts | null = null;
let observer: ResizeObserver | null = null;

function render() {
  if (!chart) return;
  const dates = [...new Set(props.rows.map((row) => row.date))].sort();
  const boatByDate = new Map<string, number>();
  const shoreByDate = new Map<string, number>();
  for (const row of props.rows) {
    if (row.source === "boat") {
      boatByDate.set(row.date, (boatByDate.get(row.date) ?? 0) + row.individuals);
    } else {
      shoreByDate.set(row.date, (shoreByDate.get(row.date) ?? 0) + row.individuals);
    }
  }
  chart.setOption({
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown) => {
        const list = Array.isArray(params) ? params : [params];
        const first = list[0] as { axisValue?: string };
        const iso = String(first?.axisValue ?? "");
        const lines = [`<strong>${iso.length === 10 ? formatDate(iso) : iso}</strong>`];
        for (const item of list as Array<{ seriesName?: string; value?: number }>) {
          if (item.value) lines.push(`${item.seriesName}: ${item.value} ejemplares`);
        }
        return lines.join("<br>");
      },
    },
    legend: { data: ["Barco", "Orilla"] },
    grid: { left: 48, right: 16, bottom: 64, top: 32 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 8 }],
    xAxis: {
      type: "category",
      data: dates,
      axisLabel: {
        formatter: (val: string) => (val.length === 10 ? formatDate(val) : val),
      },
    },
    yAxis: { type: "value", name: "Ejemplares" },
    series: [
      {
        name: "Barco",
        type: "bar",
        stack: "total",
        data: dates.map((d) => boatByDate.get(d) ?? 0),
        itemStyle: { color: "#000000", borderRadius: [4, 4, 0, 0] },
      },
      {
        name: "Orilla",
        type: "bar",
        stack: "total",
        data: dates.map((d) => shoreByDate.get(d) ?? 0),
        itemStyle: { color: "#06C167", borderRadius: [4, 4, 0, 0] },
      },
    ],
  });
}

onMounted(() => {
  if (!el.value) return;
  chart = echarts.init(el.value);
  chart.on("click", (params) => {
    const date = String(params.name ?? "");
    if (!date || date.length !== 10) return;
    const source = params.seriesName === "Orilla" ? "shore" : "boat";
    router.push({ path: "/", query: { source, date } });
  });
  render();
  observer = new ResizeObserver(() => chart?.resize());
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
