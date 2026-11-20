// Self-hosted receiver for Lumen's dead-man's switch.
//
// scripts/lumen-heartbeat.sh pings ANIMA_HEARTBEAT_URL when every liveness
// probe passes and ANIMA_HEARTBEAT_URL/fail when one does not. This Worker is
// the other end: it records each ping and posts to a webhook when the pings
// stop, when the Pi reports a failure, and when it recovers.
//
//   GET /<PING_TOKEN>          healthy ping
//   GET /<PING_TOKEN>/fail     the Pi says Lumen is not healthy
//   GET /<PING_TOKEN>/status   JSON view of the switch, for verification
//
// The state lives in one Durable Object. Its alarm is the dead-man timer: each
// ping moves the silence deadline to "last contact + grace". The same alarm
// also carries reminders and delivery retries. No cron, and no storage ids to
// configure.
//
// Failure direction: every change of condition is written to a stored outbox
// and stays there until the webhook accepts it. The alarm retries delivery, in
// order, so an outage that begins and ends while the webhook is down is still
// reported, late and marked as late.
//
// The token is in the URL path because that is the protocol the Pi script
// speaks. Do not turn on Workers invocation logs for this Worker: they record
// request URLs.
//
// This file has no imports so the same code runs under `node --test`.

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const RETRY_MS = 5 * MINUTE;
// Well under the Pi's `curl -m 10`: a ping can wait behind one webhook post.
const NOTIFY_TIMEOUT_MS = 5_000;
const MIN_TOKEN_LENGTH = 16;
const MAX_OUTBOX = 50;
const DELAY_NOTE_MS = 2 * MINUTE;
// Posts per step, so a ping never waits behind a long drain.
const FLUSH_BUDGET = 5;
// Discord rejects content over 2000 characters, and a notice that can never be
// accepted would block every notice behind it.
const MAX_CONTENT = 1900;

export function readConfig(env) {
  return {
    name: text(env.SWITCH_NAME) || "Lumen",
    graceMs: positive(env.GRACE_MINUTES, 20) * MINUTE,
    // 0 turns reminders off.
    remindMs: nonNegative(env.REMIND_HOURS, 24) * HOUR,
    mention: text(env.ALERT_MENTION),
    webhook: text(env.ALERT_WEBHOOK_URL),
    token: text(env.PING_TOKEN),
  };
}

function text(value) {
  return typeof value === "string" ? value.trim() : "";
}

function positive(value, fallback) {
  const n = Number(text(String(value ?? "")));
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function nonNegative(value, fallback) {
  const raw = text(String(value ?? ""));
  if (raw === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

// What is true at `now`. Silence outranks a reported failure: a Pi that said
// "unhealthy" and then went quiet is a bigger problem than either alone.
export function condition(state, now, graceMs) {
  if (!state.lastContactAt) return "new";
  if (now - state.lastContactAt >= graceMs) return "late";
  return state.lastKind === "fail" ? "fail" : "up";
}

function isDown(c) {
  return c === "late" || c === "fail";
}

function duration(ms) {
  const minutes = Math.max(0, Math.round(ms / MINUTE));
  if (minutes < 90) return `${minutes} min`;
  const hours = ms / HOUR;
  if (hours < 48) return `${hours.toFixed(1)} h`;
  return `${(hours / 24).toFixed(1)} d`;
}

function utc(ms) {
  return ms ? new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z") : null;
}

export function transitionMessage(cfg, from, to, state, now) {
  const n = cfg.name;
  if (to === "late") {
    return (
      `DOWN: ${n} is silent. No heartbeat for ${duration(now - state.lastContactAt)} ` +
      `(last contact ${utc(state.lastContactAt)}). The Pi is off, off the network, ` +
      `or its heartbeat timer stopped.`
    );
  }
  if (to === "fail") {
    return (
      `DOWN: ${n} reports it is not healthy. The Pi is reachable but a liveness ` +
      `probe failed. See ~/.anima/heartbeat.log on the Pi.`
    );
  }
  if (isDown(from)) {
    return `RECOVERED: ${n} is healthy again after ${duration(now - (state.since || now))} down.`;
  }
  return (
    `ARMED: first healthy heartbeat from ${n}. ` +
    `Silence longer than ${duration(cfg.graceMs)} will alert here.`
  );
}

function reminderMessage(cfg, current, state, now) {
  const what = current === "late" ? "silent" : "reporting failure";
  return `STILL DOWN: ${cfg.name} has been ${what} for ${duration(now - (state.since || now))}.`;
}

function enqueue(state, at, message) {
  state.outbox.push({ at, text: message });
  if (state.outbox.length > MAX_OUTBOX) {
    const gone = state.outbox.shift();
    state.dropped = (state.dropped || 0) + 1;
    state.droppedSince ||= gone.at;
  }
}

// Compare the condition at `now` with the one last observed and queue a notice
// for any change. Notices go to a stored outbox and are delivered in order, so
// what was observed never depends on whether the webhook was working: a
// rejected DOWN followed by a recovery is still two notices, sent late.
export function observe(state, now, cfg) {
  const current = condition(state, now, cfg.graceMs);
  if (current === "new") return current;
  const from = state.observed;
  if (current !== from) {
    enqueue(state, now, transitionMessage(cfg, from, current, state, now));
    if (!isDown(current)) state.since = null;
    else {
      // `since` marks the start of the outage and survives late <-> fail, so a
      // recovery reports the whole outage, not just its last phase.
      if (!isDown(from)) state.since = current === "late" ? state.lastContactAt : now;
      state.remindedAt = now;
    }
    state.observed = current;
  } else if (
    isDown(current) &&
    cfg.remindMs > 0 &&
    now - (state.remindedAt || now) >= cfg.remindMs &&
    // No reminders on top of notices that are not getting through.
    state.outbox.length === 0
  ) {
    enqueue(state, now, reminderMessage(cfg, current, state, now));
    state.remindedAt = now;
  }
  return current;
}

// Deliver the outbox oldest first, stopping at the first rejection or when the
// step's budget is spent. Returns "done", "failed" or "more". Delivery is
// at-least-once: a crash after a post and before the state is stored repeats it.
export async function flush(state, clock, notify) {
  let posts = 0;
  while (state.outbox.length > 0) {
    if (posts >= FLUSH_BUDGET) return "more";
    const now = clock();
    const item = state.outbox[0];
    let message = item.text;
    if (now - item.at >= DELAY_NOTE_MS) {
      message += ` (Delayed notice: this happened at ${utc(item.at)}.)`;
    }
    if (state.dropped) {
      message +=
        ` (${state.dropped} earlier notices, from ${utc(state.droppedSince)} on, were dropped ` +
        `undelivered: the state changed repeatedly while this webhook was unreachable.)`;
    }
    posts += 1;
    if (!(await notify(message))) return "failed";
    state.outbox.shift();
    state.dropped = 0;
    state.droppedSince = null;
    state.notifiedAt = now;
  }
  return "done";
}

// When the alarm should next fire, or null.
export function nextWakeup(state, now, cfg, current, outcome) {
  const wakeups = [];
  if (current === "up" || current === "fail") wakeups.push(state.lastContactAt + cfg.graceMs);
  if (isDown(current) && cfg.remindMs > 0) {
    wakeups.push((state.remindedAt || now) + cfg.remindMs);
  }
  // A deadline already due has been handled or is waiting on delivery. Waiting
  // on it again would spin; the retry interval replaces it.
  const future = wakeups.filter((t) => t > now);
  if (outcome === "failed") future.push(now + RETRY_MS);
  if (outcome === "more") future.push(now + 1000);
  if (future.length > 0) return Math.min(...future);
  return wakeups.length > 0 ? now + 1000 : null;
}

export function recordPing(state, kind, now) {
  state.lastContactAt = now;
  state.lastKind = kind;
  if (kind === "fail") state.lastFailAt = now;
  else state.lastOkAt = now;
}

// Discord answers 204 before it has stored the message unless asked to wait, so
// wait=true is forced even if the configured URL says otherwise. For any other
// host a 2xx is taken as accepted; there is nothing further to check.
export function webhookTarget(raw) {
  try {
    const url = new URL(raw);
    if (/(^|\.)discord(app)?\.com$/.test(url.hostname)) url.searchParams.set("wait", "true");
    return url.toString();
  } catch {
    return raw;
  }
}

async function postWebhook(cfg, content) {
  if (!cfg.webhook) return false;
  const full = cfg.mention ? `${cfg.mention} ${content}` : content;
  const body = JSON.stringify({
    content: full.length > MAX_CONTENT ? `${full.slice(0, MAX_CONTENT - 1)}…` : full,
  });
  try {
    const res = await fetch(webhookTarget(cfg.webhook), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      // A redirect (a login page, say) must not end in a 200 that reads as
      // delivered.
      redirect: "manual",
      signal: AbortSignal.timeout(NOTIFY_TIMEOUT_MS),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export class HeartbeatSwitch {
  constructor(ctx, env) {
    this.storage = ctx.storage;
    this.env = env;
    this.queue = Promise.resolve();
    this.inbox = [];
  }

  // One step at a time. A webhook post is a network wait, and the platform
  // delivers other events during it; without this a slow alarm could write its
  // older snapshot over a ping that had already been answered 200.
  //
  // A ping is stamped when it arrives, not when its turn comes: one that waited
  // behind a slow post must not be judged late for the wait.
  run(kind) {
    if (kind) this.inbox.push({ kind, at: Date.now() });
    const job = this.queue.then(() => this.step());
    this.queue = job.catch(() => {});
    return job;
  }

  async step() {
    const cfg = readConfig(this.env);
    const state = (await this.storage.get("state")) || {};
    state.outbox ||= [];
    const arrivals = this.inbox;
    this.inbox = [];
    for (const ping of arrivals) {
      // Judge the silence this ping ends before the ping erases it. Alarms can
      // run late; a ping arriving just past the deadline would otherwise reset
      // the clock and the breach would never be reported.
      observe(state, ping.at, cfg);
      recordPing(state, ping.kind, ping.at);
      observe(state, ping.at, cfg);
    }
    const now = Date.now();
    const current = observe(state, now, cfg);
    let outcome = "done";
    if (state.outbox.length > 0) {
      // Store the notices, with a retry alarm, before posting any of them. If
      // the object restarts mid-post the notice is still owed, not forgotten.
      await this.commit(state, nextWakeup(state, now, cfg, current, "failed"));
      outcome = await flush(state, Date.now, (m) => postWebhook(cfg, m));
    }
    await this.commit(state, nextWakeup(state, Date.now(), cfg, current, outcome));
  }

  // Alarm and state together, with no await between them, so the platform
  // commits both or neither. State stored without its alarm would be a switch
  // that cannot fire.
  async commit(state, next) {
    const alarm = next === null ? this.storage.deleteAlarm() : this.storage.setAlarm(next);
    await Promise.all([alarm, this.storage.put("state", state)]);
  }

  async alarm() {
    await this.run(null);
  }

  async fetch(request) {
    const path = new URL(request.url).pathname;
    if (path === "/ok" || path === "/fail") {
      await this.run(path.slice(1));
      return new Response("OK\n");
    }
    if (path === "/status") {
      const cfg = readConfig(this.env);
      const now = Date.now();
      const state = (await this.storage.get("state")) || {};
      const outbox = state.outbox || [];
      return Response.json({
        name: cfg.name,
        condition: condition(state, now, cfg.graceMs),
        observed: state.observed || null,
        undelivered_notices: outbox.length,
        oldest_undelivered: utc(outbox[0]?.at),
        last_ok: utc(state.lastOkAt),
        last_fail: utc(state.lastFailAt),
        last_contact: utc(state.lastContactAt),
        last_notified: utc(state.notifiedAt),
        next_alarm: utc(await this.storage.getAlarm()),
        grace_minutes: cfg.graceMs / MINUTE,
        remind_hours: cfg.remindMs / HOUR,
        now: utc(now),
      });
    }
    return new Response("not found\n", { status: 404 });
  }
}

async function sameSecret(a, b) {
  const enc = new TextEncoder();
  const [x, y] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(a)),
    crypto.subtle.digest("SHA-256", enc.encode(b)),
  ]);
  const xs = new Uint8Array(x);
  const ys = new Uint8Array(y);
  let diff = 0;
  for (let i = 0; i < xs.length; i++) diff |= xs[i] ^ ys[i];
  return diff === 0;
}

export default {
  async fetch(request, env) {
    const cfg = readConfig(env);
    // An unconfigured switch must not answer 200: the Pi would believe its
    // pings were landing while nothing could ever alert.
    if (cfg.token.length < MIN_TOKEN_LENGTH || !cfg.webhook) {
      return new Response("heartbeat receiver is not configured\n", { status: 503 });
    }

    const parts = new URL(request.url).pathname.split("/").filter(Boolean);
    const action = parts.length === 1 ? "ok" : parts[1];
    const known = parts.length >= 1 && parts.length <= 2 && ["ok", "fail", "status"].includes(action);
    if (!known || parts[1] === "ok" || !(await sameSecret(parts[0], cfg.token))) {
      return new Response("not found\n", { status: 404 });
    }
    if (!["GET", "HEAD", "POST"].includes(request.method)) {
      return new Response("method not allowed\n", { status: 405 });
    }

    const stub = env.SWITCH.get(env.SWITCH.idFromName("default"));
    return stub.fetch(`https://switch/${action}`);
  },
};
