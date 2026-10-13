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

export interface BehaviorRow {
  species: string;
  species_short: string;
  slug: string;
  color: string;
  behavior: string;
  events: number;
  individuals: number;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`No se pudo leer ${url}`);
  }
  return response.json() as Promise<T>;
}

let metaPromise: Promise<Meta> | null = null;

export function getMeta(): Promise<Meta> {
  if (!metaPromise) {
    metaPromise = getJson<Meta>("/api/meta");
  }
  return metaPromise;
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

export function getTracks(date?: string) {
  const params = date ? `?date=${date}` : "";
  return getJson<FeatureCollection>(`/api/tracks${params}`);
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

export function getBehavior(species?: string) {
  const params = species ? `?species=${encodeURIComponent(species)}` : "";
  return getJson<BehaviorRow[]>(`/api/stats/behavior${params}`);
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
