# Lumen's dead-man's switch

## Why

Every liveness surface Lumen had ran on the Mac: the Discord bridge poller, the
tailscale watchdog, the governance health watchdog. All of them share a failure
domain with the Mac, so none can report *"the Mac is asleep"* — and that silence
is indistinguishable from Lumen being fine.

In July 2026 the Pi was unplugged for ~3.5 days. The Discord bridge detected it
correctly and posted an embed nobody was looking at; the outage was found by
hand on the operator's return.

A dead-man's switch inverts the burden. Instead of something watching for a
failure, the healthy state has to keep asserting itself, and **absence is the
alarm**. That only works if the assertion leaves both machines, so the Pi pings a
third party that neither the Mac nor the Pi owns.

## What it asserts

**"Lumen is alive"** — not "the Pi has power".

A cron line that curls a URL unconditionally proves only that the box boots and
has network. It would have reported healthy through every software failure Lumen
has ever had. That is the fail-toward-healthy shape CLAUDE.md invariant 2
forbids.

So the heartbeat gates on Lumen's own **work output**, never on
`systemctl is-active` — a live PID is not work output.

Three processes make up the creature, plus one server-owned long-clock output;
the same argument says one component's work output is not the creature's:

| component | what it does | probe |
|---|---|---|
| `anima-broker` | sensors, learning | freshness of `/dev/shm/anima_state.json` |
| `anima-broker-ex` | Elixir; owns the governance check-ins | freshness of `…shadow.json` |
| `anima` | MCP server: agency learner (authoritative), metacognition, growth, drawing, display, the tool surface | `GET http://127.0.0.1:8766/health` |
| `day_summary` | long-term evidence writer | worse of writer and newest-summary age in `~/.anima/day_summaries.json` |

**The worst result decides.** The first version of this script checked only the
first envelope — so if the MCP server had died, the broker would have kept
writing, the envelope would have stayed fresh, and the switch would have pinged
green forever while most of Lumen was gone. That is the exact failure the switch
exists to prevent, and it took an adversarial review to catch it.

The server gets a functional probe rather than a file, deliberately: it either
answers or it does not, so there is no cached artifact that can go stale in a
way that reads healthy.

The day-summary writer creates an empty `day_summaries.json` containing a
`writer_started_at` marker on its first live-history tick. That marker permits a
30-minute first-boot warm-up while fewer than 100 observations exist. A missing
marker, an eligible source with no summary, or a marker older than the bounded
grace is unhealthy. Once summaries exist, both top-level `written_at` and the
newest row's evidence `date` must remain within 36 hours; the worse age wins.

It does **not** gate on governance reachability. Governance lives on the Mac, so
folding it in would page you about a Mac outage under the heading "Lumen is
dead", conflating two failures with different responses. That layer is the
bridge's job.

## Layering

| layer | detects | latency | blind to |
|---|---|---|---|
| Discord bridge (Mac) | sensor state, governance, drawings | ~10 min | anything that takes the Mac down with it |
| **This switch (Pi → cloud)** | Lumen not alive | provider grace (~15–20 min) | nothing local; only a total provider outage |

Two independent detectors, one destination. Neither can cover the other's blind
spot alone, which is the point.

## Failure direction

```
all unskipped probes healthy -> ping success
any probe stale/absent/bad   -> ping the /fail endpoint (alerts immediately)
this script dead             -> no ping at all -> alerts after the grace period
Pi has no power/network      -> no ping at all -> alerts after the grace period
```

All roads lead to you. There is no path where a broken heartbeat reads as a
healthy Lumen.

## Setup

1. Create a check at any dead-man's-switch provider (healthchecks.io has a free
   tier and is open-source). Period **5 min**, grace **15–20 min**. To avoid a
   third-party provider, deploy the receiver in this repo instead: see
   [Self-hosted receiver](#self-hosted-receiver), then continue at step 3.
2. Point its notification at somewhere that reaches a phone. Sending it to the
   same Discord you already watch is fine and keeps one destination — the
   independence that matters is in the *detector*, not the delivery channel.
3. On the Pi, add the secret ping URL to `~/.anima/anima.env`:

   ```
   ANIMA_HEARTBEAT_URL=https://hc-ping.com/<your-uuid>
   ```

4. Install and start the timer (`restore_lumen.sh` does this automatically on a
   reflash):

   ```bash
   sudo cp ~/anima-mcp/systemd/lumen-heartbeat.{service,timer} /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now lumen-heartbeat.timer
   ```

5. Verify it is actually pinging, not merely installed:

   ```bash
   systemctl list-timers lumen-heartbeat.timer
   sudo journalctl -u lumen-heartbeat.service -n 20
   tail ~/.anima/heartbeat.log     # quiet on the happy path; entries mean trouble
   ```

   Then confirm the provider shows a recent ping. **Installed is not the same as
   working** — that is the whole lesson of the July outage.

## Self-hosted receiver

`cloudflare/heartbeat-worker/` is a Cloudflare Worker that takes the place of
the provider. It speaks the same protocol the Pi script already uses (`GET
<url>` when healthy, `GET <url>/fail` when not), so nothing on the Pi changes
except the URL. It still runs off both machines, which is the property the
switch depends on; what it removes is the extra vendor.

One Durable Object holds the state, and its alarm is the timer: every ping moves
the silence deadline to *last contact + grace*. The same alarm carries the daily
reminder and delivery retries.

| event | webhook post |
|---|---|
| first healthy ping | `ARMED` |
| no ping for `GRACE_MINUTES` (default 20) | `DOWN: ... is silent` |
| a `/fail` ping | `DOWN: ... reports it is not healthy` |
| healthy ping after either | `RECOVERED` |
| still down after `REMIND_HOURS` (default 24, `0` disables) | `STILL DOWN` |
| recovered before the `DOWN` post could be delivered | `MISSED` |

A post the webhook does not accept is not recorded as sent. The receiver retries
every 5 minutes until it lands, and an outage that ends before its alert got
out is reported afterwards as `MISSED`. With either secret missing it answers 503, so
the Pi logs `provider unreachable` and does not ping into a receiver that cannot
alert.

Deploy from `cloudflare/heartbeat-worker/`:

```bash
npx wrangler login
npx wrangler deploy --domain heartbeat.example.org   # a hostname on a zone you own
openssl rand -hex 24 | npx wrangler secret put PING_TOKEN
npx wrangler secret put ALERT_WEBHOOK_URL            # e.g. a Discord channel webhook
```

The webhook receives `{"content": "<message>"}`, which is Discord's format. Set
`ALERT_MENTION` (for example `<@your-user-id>`) if the channel does not notify
on every message.

The Pi's URL is then `https://heartbeat.example.org/<PING_TOKEN>`. Treat it like
any provider ping URL: it is the secret. Two consequences:

- Leave Workers invocation logs off for this Worker (they are off unless
  `[observability]` is added to `wrangler.toml`). They record request URLs.
- If `anima.env` sets `ANIMA_HEARTBEAT_FAIL_URL`, remove it or point it at
  `<url>/fail` on the receiver. A leftover override sends failures to the old
  provider, and the receiver never hears them.

Verify with `curl https://heartbeat.example.org/<PING_TOKEN>/status`, which
shows the last contact, the announced state and the next alarm. The `ARMED`
post arriving on the phone is the proof that the delivery hop works. No `ARMED`
post means the switch is not working, whatever the timer says. A receiver that
has never been pinged does not alarm, so that first post is the only evidence
that pings are landing.

Tests: `node --test cloudflare/heartbeat-worker/test/worker.test.mjs`.

## Testing it for real without interrupting Lumen

After the provider shows a successful ping, use a stale summary fixture to
exercise the real `day_summary` gate, the provider's `/fail` endpoint, and the
phone notification. This does not stop the broker or MCP server:

```bash
set -eu
fixture="$(mktemp "$HOME/.anima/day_summaries.heartbeat-test.XXXXXX.json")"
cleanup() {
  rm -f "$fixture"
  sudo systemctl start lumen-heartbeat.timer
}
trap cleanup EXIT INT TERM

printf '%s\n' '{"written_at":"2000-01-01T00:00:00+00:00","summaries":[{"date":"2000-01-01T00:00:00+00:00"}]}' > "$fixture"
sudo systemctl stop lumen-heartbeat.timer
DAY_SUMMARY_PATH="$fixture" ~/anima-mcp/scripts/lumen-heartbeat.sh
```

Confirm all three pieces of evidence before continuing: `heartbeat.log` says
`day_summary stale` and `signalling failure`; the provider records a failure;
and the configured phone destination receives the actual notification. Then
send recovery through the same script and confirm the provider returns green:

```bash
~/anima-mcp/scripts/lumen-heartbeat.sh
```

The trap removes the fixture and restarts the timer. If notification delivery
is delayed, keep the timer stopped only long enough to inspect the provider;
run `cleanup` manually before leaving the shell. A provider dashboard event
alone does not prove the final delivery hop.

## Notes

- The URL is a **secret**: anyone holding it can suppress your alerts by pinging
  it. It lives in `anima.env`, which is deliberately excluded from backups.
- Consequence: after a reflash the URL is gone, the pings stop, and the provider
  alerts. Loud rather than silent — the correct direction. `restore_lumen.sh`
  lists `ANIMA_HEARTBEAT_URL` among the keys that came back blank.
- Unset URL → the script exits 0 and logs at most daily, so an un-provisioned Pi
  is not a crash loop. Its immutable `.heartbeat-inert` first-seen marker is
  surfaced as `outbound_heartbeat` by `/health/detailed`: healthy during a
  24-hour provisioning grace, then degraded until the URL is set. A separate
  `.heartbeat-inert.notice` file throttles logs; both clear on provisioning.
- Tunables: `ANIMA_HEARTBEAT_MAX_AGE` (default 120s), `ANIMA_HEARTBEAT_FAIL_URL`
  (defaults to `<url>/fail`), `ANIMA_SHM_PATH`, `ANIMA_HEARTBEAT_SHADOW_PATH`,
  `ANIMA_HEARTBEAT_SERVER_URL`, `DAY_SUMMARY_PATH`, `ANIMA_HISTORY_PATH`, and
  `DAY_SUMMARY_MAX_AGE` (default 129600s / 36h), plus
  `DAY_SUMMARY_BOOTSTRAP_GRACE_SECONDS` (default 1800s / 30 min). Invalid age
  values fail closed and signal the provider's `/fail` endpoint. All tunables
  may be set in `anima.env`.
- `ANIMA_HEARTBEAT_SKIP` is a comma list (`broker`, `broker_ex`, `server`,
  `day_summary`) for
  **documented rollbacks only**. Reverting the Elixir broker leaves a stale
  shadow envelope that would otherwise page forever; rolling back the summary
  writer may likewise require a temporary `day_summary` skip. Skipping a
  component is choosing not to be told about it — set it deliberately, and
  unset it when the rollback ends.

## What this still does not cover

The switch tells you Lumen stopped. It does not keep Lumen running, and it does
not remove the dependency on one person being reachable to act on the page. It
narrows the window from days to minutes; it does not make the arrangement
robust to that person being unavailable.
