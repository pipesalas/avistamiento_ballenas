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

function buildPopupNode(
  features: Array<{ properties?: Record<string, unknown> | null }>,
  kind: "sighting" | "zone" | "cell",
): HTMLElement {
  const container = document.createElement("div");
  container.className = "map-popup";
  const first = features[0]?.properties ?? {};

  if (kind === "zone") {
    const title = document.createElement("strong");
    title.textContent = `Subzona ${first.subzone ?? ""}`;
    container.appendChild(title);
    const lines = String(first.summary ?? "").split("\n").filter(Boolean);
    for (const line of lines) {
      const div = document.createElement("div");
      div.textContent = line;
      container.appendChild(div);
    }
    return container;
  }

  if (kind === "cell") {
    container.textContent = String(first.label ?? "");
    return container;
  }

  features.slice(0, 6).forEach((feature, idx) => {
    const props = feature.properties ?? {};
    const item = document.createElement("div");
    if (idx > 0) {
      item.style.borderTop = "1px solid #eee";
      item.style.marginTop = "6px";
      item.style.paddingTop = "6px";
    }
    const title = document.createElement("strong");
    title.textContent = String(props.species_short ?? "Avistamiento");
    item.appendChild(title);

    const metaLine = document.createElement("div");
    const sourceLabel = props.source === "shore" ? `Orilla (Subzona ${props.subzone ?? "—"})` : "Barco";
    metaLine.textContent = `${props.individuals ?? 0} ejemplares · ${sourceLabel}`;
    item.appendChild(metaLine);

    if (props.behavior && props.behavior !== "null") {
      const behaviorEl = document.createElement("div");
      behaviorEl.textContent = String(props.behavior);
      item.appendChild(behaviorEl);
    }
    if (props.notes && props.notes !== "null") {
      const notesEl = document.createElement("div");
      notesEl.style.color = "#545454";
      notesEl.textContent = String(props.notes);
      item.appendChild(notesEl);
    }
    const dateEl = document.createElement("div");
    dateEl.style.fontSize = "12px";
    dateEl.style.color = "#545454";
    dateEl.textContent = `${props.date ?? ""}${props.hour ? ` · ${props.hour}` : ""}`;
    item.appendChild(dateEl);
    container.appendChild(item);
  });

  if (features.length > 6) {
    const more = document.createElement("div");
    more.style.marginTop = "6px";
    more.style.fontSize = "12px";
    more.style.fontWeight = "700";
    more.textContent = `+ ${features.length - 6} registros más en este punto`;
    container.appendChild(more);
  }

  return container;
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
        "circle-stroke-color": [
          "case",
          ["==", ["get", "source"], "shore"],
          "#000000",
          "#ffffff",
        ],
        "circle-stroke-width": 1.75,
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
        const features = event.features;
        if (!features?.length || !map) return;
        new maplibregl.Popup({ maxWidth: "300px" })
          .setLngLat(event.lngLat)
          .setDOMContent(buildPopupNode(features, kind))
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
