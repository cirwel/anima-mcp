// Run: node --test cloudflare/heartbeat-worker/test/worker.test.mjs
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import worker, { HeartbeatSwitch, condition, webhookTarget } from "../src/worker.js";

const MIN = 60_000;
const HOUR = 60 * MIN;
const TOKEN = "t".repeat(32);
const T0 = Date.UTC(2026, 0, 1, 12, 0, 0);

let now;
let sent;
let webhookUp;
let realFetch;
let realNow;
let requests;
let gate;

beforeEach(() => {
  now = T0;
  sent = [];
  webhookUp = true;
  realNow = Date.now;
  realFetch = globalThis.fetch;
  Date.now = () => now;
  requests = [];
  gate = null;
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init });
    if (gate) await gate;
    if (!webhookUp) return new Response("rate limited", { status: 429 });
    sent.push(JSON.parse(init.body).content);
    return new Response(null, { status: 204 });
  };
});

afterEach(() => {
  Date.now = realNow;
  globalThis.fetch = realFetch;
});

// Let in-flight requests reach the switch. Token comparison hashes on a
// thread pool, so one turn of the event loop is not enough; the clock under
// test is the mocked Date.now, which this real timer does not move.
const settle = () => new Promise((r) => setTimeout(r, 25));

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
    object,
    stored: () => structuredClone(store.get("state")),
    // Move the clock without firing the alarm, as when the platform runs an
    // alarm late.
    skip: (ms) => void (now += ms),
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
  assert.equal(s.observed, "late");
  assert.equal(s.undelivered_notices, 1);
  assert.equal(s.oldest_undelivered, "2026-01-01T12:20:00Z");

  webhookUp = true;
  await h.advance(6 * MIN);
  s = await h.status();
  assert.equal(s.undelivered_notices, 0);
  assert.match(sent.at(-1), /^DOWN: Lumen is silent.*\(Delayed notice: this happened at 2026-01-01T12:20:00Z\.\)$/);
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
  assert.equal(condition({ lastContactAt: T0, lastKind: "fail" }, T0 + 1, g), "fail");
  assert.equal(condition({ lastContactAt: T0, lastKind: "fail" }, T0 + g, g), "late");
});

test("a /fail in the same millisecond as a healthy ping still alerts", async () => {
  const h = harness();
  await h.ping();
  await h.fail();
  assert.match(sent.at(-1), /^DOWN: Lumen reports it is not healthy/);
});

test("a ping that beats a late alarm does not erase the breach", async () => {
  const h = harness();
  await h.ping();
  h.skip(20 * MIN + 10_000); // the alarm is overdue and has not run
  await h.ping();
  assert.match(sent.at(-2), /^DOWN: Lumen is silent\. No heartbeat for 20 min/);
  assert.match(sent.at(-1), /^RECOVERED: Lumen is healthy again after 20 min down\./);
});

test("an outage that starts and ends while the webhook is down is reported late", async () => {
  const h = harness();
  await h.ping();
  await h.advance(5 * MIN);
  webhookUp = false;
  await h.fail();
  await h.advance(5 * MIN);
  await h.ping(); // recovered, webhook still down
  assert.equal(sent.length, 1);
  assert.equal((await h.status()).undelivered_notices, 2);
  webhookUp = true;
  await h.advance(6 * MIN);
  assert.match(sent.at(-2), /^DOWN: Lumen reports it is not healthy.*\(Delayed notice: this happened at 2026-01-01T12:05:00Z\.\)$/);
  assert.match(sent.at(-1), /^RECOVERED: Lumen is healthy again after 5 min down\. \(Delayed notice: this happened at 2026-01-01T12:10:00Z\.\)$/);
  assert.equal((await h.status()).undelivered_notices, 0);
});

test("a later outage does not erase an earlier undelivered one", async () => {
  const h = harness();
  await h.ping();
  webhookUp = false;
  await h.advance(5 * MIN);
  await h.fail();
  await h.advance(5 * MIN);
  await h.ping();
  await h.advance(30 * MIN); // now silent, webhook still down
  webhookUp = true;
  await h.advance(6 * MIN);
  const kinds = sent.slice(1).map((m) => m.split(":")[0] + (m.includes("silent") ? " silent" : ""));
  assert.deepEqual(kinds, ["DOWN", "RECOVERED", "DOWN silent"]);
});

test("a rejected RECOVERED does not mask the next outage", async () => {
  const h = harness();
  await h.ping();
  await h.advance(5 * MIN);
  await h.fail(); // DOWN delivered
  webhookUp = false;
  await h.advance(5 * MIN);
  await h.ping(); // RECOVERED rejected
  await h.advance(5 * MIN);
  await h.fail(); // second outage, still rejected
  webhookUp = true;
  await h.advance(6 * MIN);
  assert.deepEqual(
    sent.slice(1).map((m) => m.split(":")[0]),
    ["DOWN", "RECOVERED", "DOWN"],
  );
});

test("a ping that waited behind a slow post is judged by when it arrived", async () => {
  const h = harness({ REMIND_HOURS: "0" });
  await h.ping();
  await h.advance(10 * MIN);
  let release;
  gate = new Promise((r) => (release = r));
  const failing = h.fail(); // DOWN post in flight
  await settle();
  now = T0 + 10 * MIN + 19 * MIN + 59_000; // one second inside the grace period
  const ping = h.ping();
  await settle();
  now += 2 * MIN; // its turn comes after the deadline
  gate = null;
  release();
  await Promise.all([failing, ping]);
  assert.equal(sent.filter((m) => m.includes("silent")).length, 0);
  assert.match(sent.at(-1), /^RECOVERED: Lumen is healthy again after 20 min down\./);
  assert.equal((await h.status()).last_ok, "2026-01-01T12:29:59Z");
});

test("the outbox is bounded and says what it dropped", async () => {
  const h = harness({ REMIND_HOURS: "0" });
  await h.ping();
  webhookUp = false;
  for (let i = 0; i < 40; i++) {
    await h.advance(MIN);
    await h.fail();
    await h.advance(MIN);
    await h.ping();
  }
  assert.equal((await h.status()).undelivered_notices, 50);
  webhookUp = true;
  await h.advance(6 * MIN);
  assert.equal((await h.status()).undelivered_notices, 0);
  // The first outage began at 12:01; its DOWN was dropped, and the notice says so.
  const note = sent.filter((m) => m.includes("earlier notices"));
  assert.equal(note.length, 1);
  assert.match(note[0], /\(30 earlier notices, from 2026-01-01T12:01:00Z on, were dropped undelivered/);
});

test("a long drain is spread over steps so pings are not held up", async () => {
  const h = harness({ REMIND_HOURS: "0" });
  await h.ping();
  webhookUp = false;
  for (let i = 0; i < 6; i++) {
    await h.advance(MIN);
    await h.fail();
    await h.advance(MIN);
    await h.ping();
  }
  webhookUp = true;
  const before = requests.length;
  await h.advance(MIN);
  await h.ping(); // one step: at most five posts
  assert.equal(requests.length - before, 5);
  assert.equal((await h.status()).undelivered_notices, 7);
  await h.advance(10_000); // the alarm finishes the drain within seconds
  assert.equal((await h.status()).undelivered_notices, 0);
});

test("a notice is stored before it is posted", async () => {
  const h = harness();
  await h.ping();
  await h.advance(5 * MIN);
  let release;
  gate = new Promise((r) => (release = r));
  const failing = h.fail();
  await settle(); // DOWN post in flight; a restart here must not lose it
  const mid = h.stored();
  assert.equal(mid.observed, "fail");
  assert.equal(mid.outbox.length, 1);
  assert.equal(h.alarmAt(), T0 + 5 * MIN + 5 * MIN); // retry alarm already set
  gate = null;
  release();
  await failing;
  assert.equal(h.stored().outbox.length, 0);
});

test("an over-long notice is truncated, not left to block the queue", async () => {
  const h = harness({ SWITCH_NAME: "x".repeat(3000) });
  await h.ping();
  assert.equal(JSON.parse(requests[0].init.body).content.length, 1900);
});

test("a ping that arrives during a slow alarm post is not overwritten", async () => {
  const h = harness({ REMIND_HOURS: "0" });
  await h.ping();
  now = T0 + 20 * MIN;
  let release;
  gate = new Promise((r) => (release = r));
  const alarm = h.object.alarm(); // DOWN post is now in flight
  await settle();
  now += 1000;
  const ping = h.ping(); // must queue behind the alarm, not interleave
  await settle();
  gate = null;
  release();
  await Promise.all([alarm, ping]);
  const s = await h.status();
  assert.equal(s.condition, "up");
  assert.equal(s.observed, "up");
  assert.equal(s.last_ok, "2026-01-01T12:20:01Z");
  assert.equal(h.alarmAt(), T0 + 20 * MIN + 1000 + 20 * MIN);
  assert.match(sent.at(-1), /^RECOVERED/);
});

test("a rejected reminder retries every five minutes, not every second", async () => {
  const h = harness();
  await h.ping();
  await h.advance(21 * MIN); // DOWN sent
  webhookUp = false;
  const before = requests.length;
  await h.advance(24 * HOUR + HOUR);
  const attempts = requests.length - before;
  assert.ok(attempts >= 10 && attempts <= 14, `attempts=${attempts}`);
});

test("a dead webhook costs a ping at most one post", async () => {
  const h = harness();
  await h.ping();
  h.skip(30 * MIN);
  webhookUp = false;
  const before = requests.length;
  await h.ping();
  assert.equal(requests.length - before, 1);
});

test("Discord webhooks are asked to confirm storage; redirects are not followed", async () => {
  assert.equal(
    webhookTarget("https://discord.com/api/webhooks/1/abc"),
    "https://discord.com/api/webhooks/1/abc?wait=true",
  );
  assert.equal(
    webhookTarget("https://discord.com/api/webhooks/1/abc?wait=false"),
    "https://discord.com/api/webhooks/1/abc?wait=true",
  );
  assert.equal(webhookTarget("https://hooks.example.org/x"), "https://hooks.example.org/x");
  assert.equal(webhookTarget("https://notdiscord.com/x"), "https://notdiscord.com/x");
  const h = harness();
  await h.ping();
  assert.equal(requests[0].init.redirect, "manual");
});
