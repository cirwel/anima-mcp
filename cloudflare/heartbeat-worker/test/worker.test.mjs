// Run: node --test cloudflare/heartbeat-worker/test/worker.test.mjs
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import worker, { HeartbeatSwitch, condition } from "../src/worker.js";

const MIN = 60_000;
const HOUR = 60 * MIN;
const TOKEN = "t".repeat(32);
const T0 = Date.UTC(2026, 0, 1, 12, 0, 0);

let now;
let sent;
let webhookUp;
let realFetch;
let realNow;

beforeEach(() => {
  now = T0;
  sent = [];
  webhookUp = true;
  realNow = Date.now;
  realFetch = globalThis.fetch;
  Date.now = () => now;
  globalThis.fetch = async (_url, init) => {
    if (!webhookUp) return new Response("rate limited", { status: 429 });
    sent.push(JSON.parse(init.body).content);
    return new Response(null, { status: 204 });
  };
});

afterEach(() => {
  Date.now = realNow;
  globalThis.fetch = realFetch;
});

function harness(vars = {}) {
  const store = new Map();
  let alarmAt = null;
  const storage = {
    get: async (k) => (store.has(k) ? structuredClone(store.get(k)) : undefined),
    put: async (k, v) => void store.set(k, structuredClone(v)),
    setAlarm: async (t) => void (alarmAt = t),
    deleteAlarm: async () => void (alarmAt = null),
    getAlarm: async () => alarmAt,
  };
  const env = { PING_TOKEN: TOKEN, ALERT_WEBHOOK_URL: "https://hook.invalid/x", ...vars };
  const object = new HeartbeatSwitch({ storage }, env);
  env.SWITCH = {
    idFromName: (n) => n,
    get: () => ({ fetch: (url) => object.fetch(new Request(url)) }),
  };
  return {
    env,
    hit: (path, init) => worker.fetch(new Request(`https://hb.invalid${path}`, init), env),
    ping: () => worker.fetch(new Request(`https://hb.invalid/${TOKEN}`), env),
    fail: () => worker.fetch(new Request(`https://hb.invalid/${TOKEN}/fail`), env),
    status: async () =>
      (await worker.fetch(new Request(`https://hb.invalid/${TOKEN}/status`), env)).json(),
    alarmAt: () => alarmAt,
    // Advance the clock, firing the alarm each time it comes due, as the
    // platform would.
    async advance(ms) {
      const end = now + ms;
      while (alarmAt !== null && alarmAt <= end) {
        now = alarmAt;
        alarmAt = null;
        await object.alarm();
      }
      now = end;
    },
  };
}

test("first healthy ping arms the switch and proves the delivery hop", async () => {
  const h = harness();
  assert.equal((await h.ping()).status, 200);
  assert.equal(sent.length, 1);
  assert.match(sent[0], /^ARMED: first healthy heartbeat from Lumen/);
  assert.equal(h.alarmAt(), T0 + 20 * MIN);
});

test("steady pings stay quiet", async () => {
  const h = harness();
  for (let i = 0; i < 50; i++) {
    await h.ping();
    await h.advance(5 * MIN);
  }
  assert.equal(sent.length, 1); // only ARMED
});

test("silence past the grace period alerts once, then reminds daily", async () => {
  const h = harness();
  await h.ping();
  await h.advance(19 * MIN);
  assert.equal(sent.length, 1);
  await h.advance(2 * MIN);
  assert.equal(sent.length, 2);
  assert.match(sent[1], /^DOWN: Lumen is silent\. No heartbeat for 20 min \(last contact 2026-01-01T12:00:00Z\)/);

  await h.advance(23 * HOUR);
  assert.equal(sent.length, 2);
  await h.advance(2 * HOUR);
  assert.equal(sent.length, 3);
  assert.match(sent[2], /^STILL DOWN: Lumen has been silent for/);
});

test("recovery reports the whole outage", async () => {
  const h = harness();
  await h.ping();
  await h.advance(3 * HOUR);
  await h.ping();
  assert.match(sent.at(-1), /^RECOVERED: Lumen is healthy again after 3\.0 h down\./);
  assert.equal((await h.status()).condition, "up");
});

test("a /fail ping alerts immediately and does not count as healthy", async () => {
  const h = harness();
  await h.ping();
  await h.advance(5 * MIN);
  assert.equal((await h.fail()).status, 200);
  assert.match(sent.at(-1), /^DOWN: Lumen reports it is not healthy/);

  // Repeated /fail pings every 5 minutes do not re-alert.
  const count = sent.length;
  for (let i = 0; i < 12; i++) {
    await h.advance(5 * MIN);
    await h.fail();
  }
  assert.equal(sent.length, count);

  await h.advance(5 * MIN);
  await h.ping();
  assert.match(sent.at(-1), /^RECOVERED: Lumen is healthy again after 65 min down\./);
});

test("failing and then going silent raises a second alert", async () => {
  const h = harness();
  await h.ping();
  await h.fail();
  const count = sent.length;
  await h.advance(21 * MIN);
  assert.equal(sent.length, count + 1);
  assert.match(sent.at(-1), /^DOWN: Lumen is silent/);
});

test("an undelivered alert is retried, never recorded as sent", async () => {
  const h = harness();
  await h.ping();
  webhookUp = false;
  await h.advance(40 * MIN);
  let s = await h.status();
  assert.equal(s.condition, "late");
  assert.equal(s.announced, "up");
  assert.equal(s.pending_notification, true);

  webhookUp = true;
  await h.advance(6 * MIN);
  s = await h.status();
  assert.equal(s.announced, "late");
  assert.equal(s.pending_notification, false);
  assert.match(sent.at(-1), /^DOWN: Lumen is silent/);
});

test("an undelivered ARMED notice is retried on the next ping", async () => {
  const h = harness();
  webhookUp = false;
  assert.equal((await h.ping()).status, 200);
  assert.equal(sent.length, 0);
  webhookUp = true;
  await h.advance(5 * MIN);
  await h.ping();
  assert.match(sent[0], /^ARMED/);
});

test("reminders can be turned off", async () => {
  const h = harness({ REMIND_HOURS: "0" });
  await h.ping();
  await h.advance(72 * HOUR);
  assert.equal(sent.filter((m) => m.startsWith("STILL DOWN")).length, 0);
  assert.equal(sent.filter((m) => m.startsWith("DOWN")).length, 1);
  assert.equal(h.alarmAt(), null);
});

test("a switch that has never been pinged does not alarm", async () => {
  const h = harness();
  await h.advance(72 * HOUR);
  assert.equal(sent.length, 0);
  assert.equal((await h.status()).condition, "new");
});

test("the mention prefix and custom name are applied", async () => {
  const h = harness({ ALERT_MENTION: "<@1>", SWITCH_NAME: "Pi" });
  await h.ping();
  assert.match(sent[0], /^<@1> ARMED: first healthy heartbeat from Pi/);
});

test("wrong token, extra path segments and odd actions are 404", async () => {
  const h = harness();
  for (const path of ["/", "/wrong", `/${TOKEN}x`, `/${TOKEN}/ok`, `/${TOKEN}/nope`, `/${TOKEN}/fail/x`, "/status"]) {
    assert.equal((await h.hit(path)).status, 404, path);
  }
  assert.equal(sent.length, 0);
  assert.equal((await h.hit(`/${TOKEN}`, { method: "DELETE" })).status, 405);
});

test("an unconfigured receiver answers 503 so the Pi's curl -f fails", async () => {
  for (const vars of [{ PING_TOKEN: "" }, { PING_TOKEN: "short" }, { ALERT_WEBHOOK_URL: "" }]) {
    const h = harness(vars);
    assert.equal((await h.hit(`/${h.env.PING_TOKEN || "x"}`)).status, 503);
  }
});

test("invalid tunables fall back to the defaults", async () => {
  const h = harness({ GRACE_MINUTES: "soon", REMIND_HOURS: "-3" });
  const s = await h.status();
  assert.equal(s.grace_minutes, 20);
  assert.equal(s.remind_hours, 24);
});

test("condition: silence outranks a reported failure", () => {
  const g = 20 * MIN;
  assert.equal(condition({}, T0, g), "new");
  assert.equal(condition({ lastContactAt: T0, lastOkAt: T0 }, T0 + g - 1, g), "up");
  assert.equal(condition({ lastContactAt: T0, lastOkAt: T0 }, T0 + g, g), "late");
  assert.equal(condition({ lastContactAt: T0, lastFailAt: T0, lastOkAt: T0 - 1 }, T0 + 1, g), "fail");
  assert.equal(condition({ lastContactAt: T0, lastFailAt: T0, lastOkAt: T0 - 1 }, T0 + g, g), "late");
});
