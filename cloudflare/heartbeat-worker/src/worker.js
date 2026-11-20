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
// The token is in the URL path because that is the protocol the Pi script
// speaks. Do not turn on Workers invocation logs for this Worker: they record
// request URLs.
//
// Failure direction: a notification that cannot be delivered is never recorded
// as delivered. The announced status stays behind the real condition and the
// alarm retries until the webhook accepts the post. An outage whose alert never
// got out before the recovery is still reported, afterwards, as MISSED.
//
// This file has no imports so the same code runs under `node --test`.

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const RETRY_MS = 5 * MINUTE;
// Well under the Pi's `curl -m 10`: a ping can wait behind one webhook post.
const NOTIFY_TIMEOUT_MS = 5_000;
const MIN_TOKEN_LENGTH = 16;

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

// What is true right now, as opposed to `state.status`, which is what the
// webhook was last told. Silence outranks a reported failure: a Pi that said
// "unhealthy" and then went quiet is a bigger problem than either alone.
export function condition(state, now, graceMs) {
  if (!state.lastContactAt) return "new";
  if (now - state.lastContactAt >= graceMs) return "late";
  return state.lastKind === "fail" ? "fail" : "up";
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
  if (from === "late" || from === "fail") {
    return `RECOVERED: ${n} is healthy again after ${duration(now - (state.since || now))} down.`;
  }
  return (
    `ARMED: first healthy heartbeat from ${n}. ` +
    `Silence longer than ${duration(cfg.graceMs)} will alert here.`
  );
}

function missedMessage(cfg, state) {
  const { kind, since } = state.unreported;
  const what = kind === "late" ? "silent" : "reporting failure";
  return (
    `MISSED: ${cfg.name} was ${what} for ${duration((state.lastOkAt || since) - since)} ` +
    `from ${utc(since)}. The alert could not be delivered at the time. It is healthy now.`
  );
}

function reminderMessage(cfg, current, state, now) {
  const what = current === "late" ? "silent" : "reporting failure";
  return `STILL DOWN: ${cfg.name} has been ${what} for ${duration(now - (state.since || now))}.`;
}

// Bring the announced status in line with the real condition, sending whatever
// that requires, and return when the alarm should next fire (or null).
export async function reconcile(state, now, cfg, notify) {
  const current = condition(state, now, cfg.graceMs);
  let pending = false;

  if (current !== "new" && current !== state.status) {
    const message = transitionMessage(cfg, state.status, current, state, now);
    if (await notify(message)) {
      // `since` marks the start of the outage and survives late <-> fail, so a
      // recovery reports the whole outage, not just its last phase.
      if (current === "up") state.since = null;
      else if (!state.since || state.status === "up" || !state.status) {
        state.since = current === "late" ? state.lastContactAt : now;
      }
      if (current !== "up") state.unreported = null;
      state.status = current;
      state.notifiedAt = now;
    } else {
      pending = true;
      // Remember the outage itself, not just that a post failed: if Lumen
      // recovers before the retry lands, condition and announced status are
      // both "up" again and there would be nothing left to send.
      if (current !== "up" && state.status !== "late" && state.status !== "fail" && !state.unreported) {
        state.unreported = { kind: current, since: current === "late" ? state.lastContactAt : now };
      }
    }
  } else if (current === "up" && state.unreported) {
    if (await notify(missedMessage(cfg, state))) state.unreported = null;
    else pending = true;
  } else if (
    (current === "late" || current === "fail") &&
    cfg.remindMs > 0 &&
    now - (state.notifiedAt || 0) >= cfg.remindMs
  ) {
    if (await notify(reminderMessage(cfg, current, state, now))) state.notifiedAt = now;
    else pending = true;
  }

  state.pendingNotify = pending;

  const wakeups = [];
  if (current === "up" || current === "fail") wakeups.push(state.lastContactAt + cfg.graceMs);
  if ((state.status === "late" || state.status === "fail") && cfg.remindMs > 0) {
    wakeups.push((state.notifiedAt || now) + cfg.remindMs);
  }
  // A deadline that is already due is what the failed post was for. Waiting on
  // it again would retry immediately, forever; the retry interval replaces it.
  const future = wakeups.filter((t) => t > now);
  if (pending) future.push(now + RETRY_MS);
  if (future.length > 0) return Math.min(...future);
  return wakeups.length > 0 ? now + 1000 : null;
}

export function recordPing(state, kind, now) {
  state.lastContactAt = now;
  state.lastKind = kind;
  if (kind === "fail") state.lastFailAt = now;
  else state.lastOkAt = now;
}

// Discord answers 204 before it has stored the message unless asked to wait.
export function webhookTarget(raw) {
  try {
    const url = new URL(raw);
    const discord = /(^|\.)discord(app)?\.com$/.test(url.hostname);
    if (discord && !url.searchParams.has("wait")) url.searchParams.set("wait", "true");
    return url.toString();
  } catch {
    return raw;
  }
}

async function postWebhook(cfg, content) {
  if (!cfg.webhook) return false;
  const body = JSON.stringify({ content: cfg.mention ? `${cfg.mention} ${content}` : content });
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
  }

  // One step at a time. A webhook post is a network wait, and the platform
  // delivers other events during it; without this a slow alarm could write its
  // older snapshot over a ping that had already been answered 200.
  run(kind) {
    const job = this.queue.then(() => this.step(kind));
    this.queue = job.catch(() => {});
    return job;
  }

  async step(kind) {
    const cfg = readConfig(this.env);
    const now = Date.now();
    const state = (await this.storage.get("state")) || {};
    // At most one failed post per step, so a dead webhook costs one timeout.
    let webhookDown = false;
    const notify = async (message) => {
      if (webhookDown) return false;
      const ok = await postWebhook(cfg, message);
      webhookDown = !ok;
      return ok;
    };
    if (kind) {
      // Judge the silence that ends with this ping before the ping erases it.
      // Alarms can run late; a ping arriving just past the deadline would
      // otherwise reset the clock and the breach would never be reported.
      if (state.lastContactAt) await reconcile(state, now, cfg, notify);
      recordPing(state, kind, now);
    }
    const next = await reconcile(state, now, cfg, notify);
    await this.storage.put("state", state);
    if (next === null) await this.storage.deleteAlarm();
    else await this.storage.setAlarm(next);
    return { state, cfg, now, next };
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
      return Response.json({
        name: cfg.name,
        condition: condition(state, now, cfg.graceMs),
        announced: state.status || null,
        pending_notification: Boolean(state.pendingNotify),
        unreported_outage: state.unreported
          ? { kind: state.unreported.kind, since: utc(state.unreported.since) }
          : null,
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
