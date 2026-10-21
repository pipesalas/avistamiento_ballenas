import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { afterEach, mock, test } from "node:test";
import ts from "typescript";

// Use the project's compiler so tests also run on Node builds without TS support.
const source = readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { getMeta, getSightings } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);

afterEach(() => mock.restoreAll());

test("concurrent identical requests share one fetch", async () => {
  const fetch = mock.method(globalThis, "fetch", async () => Response.json({ features: [] }));
  const [first, second] = await Promise.all([
    getSightings({ source: "boat" }),
    getSightings({ source: "boat" }),
  ]);
  assert.equal(fetch.mock.callCount(), 1);
  assert.deepEqual(first, second);
});

test("different filters do not share a response", async () => {
  const fetch = mock.method(globalThis, "fetch", async (url) => Response.json({ url }));
  const [boat, shore] = await Promise.all([
    getSightings({ source: "boat" }),
    getSightings({ source: "shore" }),
  ]);
  assert.equal(fetch.mock.callCount(), 2);
  assert.notDeepEqual(boat, shore);
});

test("failed metadata requests can be retried", async () => {
  let fail = true;
  const fetch = mock.method(globalThis, "fetch", async () =>
    fail ? new Response("Unavailable", { status: 503 }) : Response.json({ species: [] }),
  );
  await assert.rejects(getMeta(), /No se pudo leer/);
  fail = false;
  assert.deepEqual(await getMeta(), { species: [] });
  assert.equal(fetch.mock.callCount(), 2);
});

test("completed requests defer freshness to the HTTP cache", async () => {
  let version = 1;
  mock.method(globalThis, "fetch", async () => Response.json({ version }));
  assert.deepEqual(await getMeta(), { version: 1 });
  version = 2;
  assert.deepEqual(await getMeta(), { version: 2 });
});
