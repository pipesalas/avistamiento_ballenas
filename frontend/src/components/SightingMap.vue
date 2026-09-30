<script setup lang="ts">
import maplibregl, { type GeoJSONSource, type LngLatBounds, type Map } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { FeatureCollection } from "../api";

const props = defineProps<{
  environment: FeatureCollection | null;
  tracks: FeatureCollection | null;
  zones: FeatureCollection | null;
  sightings: FeatureCollection | null;
  fit?: boolean;
}>();

const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
const el = ref<HTMLDivElement | null>(null);
let map: Map | null = null;
let ready = false;

function asGeoJson(data: FeatureCollection | null) {
  return (data ?? empty) as Parameters<GeoJSONSource["setData"]>[0];
}

function setSource(id: string, data: FeatureCollection | null) {
  const source = map?.getSource(id) as GeoJSONSource | undefined;
  source?.setData(asGeoJson(data));
}

function extendBounds(bounds: LngLatBounds, coordinates: unknown) {
  if (!Array.isArray(coordinates)) return;
  if (typeof coordinates[0] === "number" && typeof coordinates[1] === "number") {
    bounds.extend([coordinates[0], coordinates[1]]);
    return;
  }
  for (const item of coordinates) extendBounds(bounds, item);
}

function focusOnObservations() {
  if (!map || !ready || !props.fit) return;
  const bounds = new maplibregl.LngLatBounds();
  const features = [...(props.sightings?.features ?? []), ...(props.zones?.features ?? [])];
  for (const feature of features) extendBounds(bounds, feature.geometry?.coordinates);
  if (bounds.isEmpty()) return;
  map.fitBounds(bounds, {
    padding: { top: 24, left: 312, right: 48, bottom: 24 },
    maxZoom: 12,
    duration: 450,
  });
}

function sync() {
  if (!ready) return;
  setSource("environment", props.environment);
  setSource("tracks", props.tracks);
  setSource("zones", props.zones);
  setSource("sightings", props.sightings);
  void nextTick(() => {
    map?.resize();
    focusOnObservations();
  });
}

function popupHtml(properties: Record<string, unknown>, kind: "sighting" | "zone" | "cell") {
  if (kind === "zone") {
    const summary = String(properties.summary ?? "").split("\n").join("<br>");
    return `<strong>Subzona ${properties.subzone}</strong><br>${summary}`;
  }
  if (kind === "cell") {
    return String(properties.label ?? "");
  }
  const behavior = properties.behavior && properties.behavior !== "null" ? `<br>${properties.behavior}` : "";
  const notes = properties.notes && properties.notes !== "null" ? `<br>${properties.notes}` : "";
  return `<strong>${properties.species_short}</strong><br>${properties.individuals} ejemplares${behavior}${notes}<br>${properties.date}`;
}

onMounted(() => {
  if (!el.value) return;
  map = new maplibregl.Map({
    container: el.value,
    style: "https://tiles.openfreemap.org/styles/positron",
    center: [-73.75, -39.92],
    zoom: 11,
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");
  map.on("load", () => {
    if (!map) return;
    map.addSource("environment", { type: "geojson", data: asGeoJson(null) });
    map.addLayer({
      id: "environment",
      type: "fill",
      source: "environment",
      paint: { "fill-color": ["get", "color"], "fill-opacity": 0.55 },
    });
    map.addSource("zones", { type: "geojson", data: asGeoJson(null) });
    map.addLayer({
      id: "zones",
      type: "fill",
      source: "zones",
      paint: { "fill-color": ["get", "color"], "fill-opacity": 0.35 },
    });
    map.addLayer({
      id: "zones-line",
      type: "line",
      source: "zones",
      paint: { "line-color": ["get", "color"], "line-width": 2 },
    });
    map.addSource("tracks", { type: "geojson", data: asGeoJson(null) });
    map.addLayer({
      id: "tracks",
      type: "line",
      source: "tracks",
      paint: { "line-color": "#334155", "line-width": 3 },
    });
    map.addSource("sightings", { type: "geojson", data: asGeoJson(null) });
    map.addLayer({
      id: "sightings",
      type: "circle",
      source: "sightings",
      paint: {
        "circle-color": ["get", "color"],
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 1.5,
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["ln", ["+", ["coalesce", ["get", "individuals"], 0], 1]],
          0,
          7,
          3,
          22,
        ],
      },
    });

    const open = (layer: string, kind: "sighting" | "zone" | "cell") => {
      map?.on("click", layer, (event) => {
        const feature = event.features?.[0];
        if (!feature || !map) return;
        new maplibregl.Popup()
          .setLngLat(event.lngLat)
          .setHTML(popupHtml(feature.properties ?? {}, kind))
          .addTo(map);
      });
      map?.on("mouseenter", layer, () => {
        if (map) map.getCanvas().style.cursor = "pointer";
      });
      map?.on("mouseleave", layer, () => {
        if (map) map.getCanvas().style.cursor = "";
      });
    };
    open("sightings", "sighting");
    open("zones", "zone");
    open("environment", "cell");
    ready = true;
    sync();
  });
});

watch(() => [props.environment, props.tracks, props.zones, props.sightings, props.fit], sync);
onBeforeUnmount(() => {
  map?.remove();
  map = null;
});
</script>

<template>
  <div ref="el" class="map"></div>
</template>
