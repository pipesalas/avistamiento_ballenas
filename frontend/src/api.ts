export interface Species {
  name: string;
  short: string;
  slug: string;
  color: string;
  boat_events?: number;
  boat_individuals?: number;
  shore_events?: number;
  shore_individuals?: number;
}

export interface Meta {
  project: { title: string; caption: string };
  species: Species[];
  behaviors: string[];
  subzones: { id: number; color: string }[];
  sightings: {
    start: string;
    end: string;
    dates: string[];
    boat_dates: string[];
    shore_dates: string[];
    boat_count: number;
    shore_count: number;
    boat_individuals?: number;
    shore_individuals?: number;
  };
  environment: {
    start: string;
    end: string;
    depths: number[];
    surface_depth: number;
    variables: { id: string; label: string; unit: string; ramp?: string[] }[];
  };
}

export interface FeatureCollection {
  type: "FeatureCollection";
  properties?: {
    variable?: string;
    date?: string;
    depth?: string;
    unit?: string;
    min?: number | null;
    max?: number | null;
  };
  features: Array<{
    type: "Feature";
    geometry: { type: string; coordinates: unknown };
    properties?: Record<string, unknown> | null;
  }>;
}

export interface Photo {
  url: string;
  thumbnail_url?: string;
  date: string;
  source: string;
  caption: string;
  species: string | null;
  species_short: string | null;
  slug: string | null;
}

export interface SeasonRow {
  year: number;
  month: number;
  source: string;
  events: number;
  individuals: number;
}

export interface TimelineRow {
  date: string;
  source: string;
  events: number;
  individuals: number;
}

export interface BehaviorRow {
  species: string;
  species_short: string;
  slug: string;
  color: string;
  behavior: string;
  events: number;
  individuals: number;
}

const pendingRequests = new Map<string, Promise<unknown>>();

function getJson<T>(url: string): Promise<T> {
  const pending = pendingRequests.get(url);
  if (pending) return pending as Promise<T>;

  // Share concurrent requests. HTTP Cache-Control handles subsequent reads
  // without keeping another, potentially stale copy of every dataset in JS.
  const request = fetch(url)
    .then((response) => {
      if (!response.ok) throw new Error(`No se pudo leer ${url}`);
      return response.json() as Promise<T>;
    })
    .finally(() => pendingRequests.delete(url));
  pendingRequests.set(url, request);
  return request;
}

export function getMeta(): Promise<Meta> {
  return getJson<Meta>("/api/meta");
}

export function getEnvironment(variable: string, date: string, depth: string) {
  const params = new URLSearchParams({ variable, date, depth });
  return getJson<FeatureCollection>(`/api/environment?${params}`);
}

export function getSightings(query: Record<string, string>) {
  const params = new URLSearchParams(query);
  return getJson<FeatureCollection>(`/api/sightings?${params}`);
}

export function getShoreZones(query: Record<string, string>) {
  const params = new URLSearchParams(query);
  return getJson<FeatureCollection>(`/api/shore-zones?${params}`);
}

export async function getTracks(date?: string): Promise<FeatureCollection> {
  const all = await getJson<FeatureCollection>("/api/tracks");
  if (!date) return all;
  return {
    type: "FeatureCollection",
    features: all.features.filter((feature) => feature.properties?.date === date),
  };
}

export function getPhotos(query: Record<string, string>) {
  const params = new URLSearchParams(query);
  return getJson<Photo[]>(`/api/photos?${params}`);
}

export function getSpeciesStats(source = "both") {
  return getJson<Species[]>(`/api/stats/species?source=${source}`);
}

export function getSeasonality(source = "both", species?: string) {
  const params = new URLSearchParams({ source });
  if (species) params.set("species", species);
  return getJson<SeasonRow[]>(`/api/stats/seasonality?${params}`);
}

export function getTimeline(source = "both", grain: "day" | "month" = "day", species?: string) {
  const params = new URLSearchParams({ source, grain });
  if (species) params.set("species", species);
  return getJson<TimelineRow[]>(`/api/stats/timeline?${params}`);
}

export function sightingsCsvUrl(query: Record<string, string> = {}): string {
  const params = new URLSearchParams(query);
  const qs = params.toString();
  return `/api/sightings.csv${qs ? `?${qs}` : ""}`;
}

export function getBehavior(species?: string, source = "both") {
  const params = new URLSearchParams({ source });
  if (species) params.set("species", species);
  return getJson<BehaviorRow[]>(`/api/stats/behavior?${params}`);
}

export function shiftDate(iso: string, days: number): string {
  const [year, month, day] = iso.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  date.setDate(date.getDate() + days);
  const monthText = String(date.getMonth() + 1).padStart(2, "0");
  const dayText = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${monthText}-${dayText}`;
}

export function daysBetween(start: string, end: string): number {
  const [ys, ms, ds] = start.split("-").map(Number);
  const [ye, me, de] = end.split("-").map(Number);
  const a = new Date(ys, ms - 1, ds).getTime();
  const b = new Date(ye, me - 1, de).getTime();
  return Math.round((b - a) / 86400000);
}

export function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-");
  return `${day}-${month}-${year}`;
}
