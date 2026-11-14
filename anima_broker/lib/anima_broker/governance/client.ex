defmodule AnimaBroker.Governance.Client do
  @moduledoc """
  UNITARES governance check-in client for the Elixir broker. The historical
  Phase-2 shadow-soak plan remains in
  docs/plans/2026-07-01-phase2-governance-seam-addendum.md, Option A, but the
  deployed path has completed that cutover.

  Every `interval_ms` (default 180s, matching the Python bridge) it reads the
  LIVE envelope for `body_anima`/readings, builds the body EISV projection,
  and checks in over the fleet's
  REST tool bridge (`POST /v1/tools/call`, the pattern every BEAM client uses
  — no MCP handshake needed). The decision is written into the SHADOW
  envelope's `"governance"` slice. Python passthrough republishes that slice
  into the live envelope, and the server consumes it; this is no longer an
  inert soak path.

  Identity (operator decision, 2026-07-01): SCRATCH by default — the client
  onboards its own governance identity (`force_new`, parent chained across
  restarts via an id file, dispatch_beam pattern) and echoes the returned
  `client_session_id` on every write, which is what satisfies the strict
  gate (a transport-injected CSID does not). For the short pre-cutover
  window, set `:gov_client_session_id` (env `ANIMA_GOV_EX_CSID`) to Lumen's
  substrate CSID literal to exercise the real handshake — that mode skips
  onboard entirely. The CSID must be a key with a live PG session row
  (`core.sessions`); the row renews +24h on every check-in, which IS the
  binding's durability (the server writes it at onboard via
  `bound_via=onboard_stable_session`).

  Binding durability + refusal detection (addendum amendment 2026-07-02,
  anima-mcp #97/#98/#99/#100 — cutover REQUIREMENTS):

    * a typed strict refusal (`status=identity_required`,
      `error_code=SESSION_ERROR`, `error_category=auth_error`) carries
      NEITHER `success:false` NOR an `action`, so an unguarded parse writes
      it as a silent default-"proceed" — exactly how canonical Lumen stayed
      governance-dark ~3 days after the 2026-06-30 Redis wipe. This client
      classifies it distinctly and NEVER writes the governance slice for it:
      letting `governance_at` go stale is the honest alarm channel.

    * recovery anchor: HARVEST `{agent_uuid, continuity_token}` while the
      binding is healthy (at onboard, refreshed daily via `identity()`),
      SPEND it on an `identity(resume=true)` verification when a check-in is
      identity-refused (rate-limited, uuid-mismatch-refusing). Per the
      settled ontology (3 live acceptance tests, 2026-07-03): a Redis-only
      wipe self-heals server-side via the PG session row with zero client
      action; `identity(resume=true)` RESOLVES but never WRITES a binding —
      so reaching the spend path means BOTH stores lost the binding, which
      is operator territory BY DESIGN. The client verifies its identity,
      adopts the server-canonical session key, and logs OPERATOR RECOVERY
      REQUIRED naming the runbook. It must alarm loudly — never silently
      local-fallback forever.

    * the anchor/id file (`ANIMA_GOV_EX_ID_FILE`) is the operator lever for
      re-pointing the echoed key. In fixed-CSID (substrate) mode a
      `"mode":"substrate"` anchor's canonical key is preferred over the env
      literal (#99: echo what the server actually bound); delete the file to
      restart from the env-declared key. A `"mode":"scratch"` anchor is
      ignored in substrate mode so a soak identity can never shadow the
      operator's declared substrate key.

  Acceptance (pre-cutover, per the amended addendum): delete ONLY the Redis
  session key while the client runs → the next check-in must land unaided
  (PATH2, PG row). Delete both stores → the client must log OPERATOR
  RECOVERY REQUIRED (runbook: `unitares scripts/ops/rebind-resident-session.sh
  <uuid> <key>`), not silently proceed.

  Failure posture (lessons imported from the fleet):
    * circuit breaker matching the Python bridge — 2 consecutive failures
      open it, backoff 15s doubling to 120s, any success resets;
    * onboard runs off the init path with its own retry backoff (5s→60s) and
      a 15s timeout — a single short timeout once left a governance feed
      dark for 9.5h (dispatch_beam lesson);
    * `AGENT_PAUSED` is classified distinctly, never swallowed as a generic
      tool error (Sentinel lesson: a paused resident stayed dark ~18h).
  """
  use GenServer
  require Logger

  alias AnimaBroker.Governance.{EisvMapper, LiveState}

  @onboard_timeout_ms 15_000
  @checkin_timeout_ms 20_000
  @identity_timeout_ms 15_000
  @onboard_retry_base_ms 5_000
  @onboard_retry_cap_ms 60_000
  @breaker_threshold 2
  @breaker_base_s 15
  @breaker_cap_s 120
  @reanchor_cooldown_ms 600_000
  @anchor_refresh_s 86_400

  @operator_runbook "unitares scripts/ops/rebind-resident-session.sh"

  def start_link(opts),
    do: GenServer.start_link(__MODULE__, opts, name: opts[:name] || __MODULE__)

  @impl true
  def init(opts) do
    state = %{
      url: opts[:url] || Application.get_env(:anima_broker, :governance_tools_url),
      interval_ms:
        opts[:interval_ms] || Application.get_env(:anima_broker, :governance_interval_ms, 180_000),
      http_post: opts[:http_post] || (&httpc_post/4),
      id_file: opts[:id_file] || Application.get_env(:anima_broker, :gov_identity_file),
      fixed_csid: opts[:fixed_csid] || Application.get_env(:anima_broker, :gov_client_session_id),
      live_state_opts: opts[:live_state_opts] || [],
      identity: nil,
      anchor: nil,
      reanchor_last_ms: nil,
      prev_anima: nil,
      prev_readings: nil,
      failures: 0,
      breaker_backoff_s: @breaker_base_s,
      # nil = breaker closed. NEVER initialize to 0: BEAM monotonic time has an
      # arbitrary (usually negative) epoch, so `now < 0` would be true at boot
      # and silently skip every check-in.
      blocked_until_ms: nil,
      onboard_retry_ms: @onboard_retry_base_ms
    }

    cond do
      state.url == nil ->
        :ignore

      state.fixed_csid ->
        # Pre-cutover window: real substrate identity, no onboard. A prior
        # substrate anchor's canonical key beats the env literal (#99); a
        # scratch anchor from the soak must never shadow the operator's key.
        anchor = load_anchor(state.id_file)
        anchor = if anchor && anchor["mode"] == "substrate", do: anchor, else: nil
        csid = (anchor && anchor["client_session_id"]) || state.fixed_csid

        identity = %{
          csid: csid,
          agent_uuid: anchor && anchor["agent_uuid"],
          mode: :substrate
        }

        schedule_checkin(state.interval_ms)
        {:ok, %{state | identity: identity, anchor: anchor}}

      true ->
        send(self(), :onboard)
        {:ok, state}
    end
  end

  @impl true
  def handle_info(:onboard, state) do
    parent = load_prior_uuid(state.id_file)

    args =
      %{
        "force_new" => true,
        "spawn_reason" => "elixir_broker_shadow_soak",
        "name" => "lumen-broker-ex-shadow",
        "response_mode" => "minimal"
      }
      |> maybe_put("parent_agent_id", parent)

    case call_tool(state, "onboard", args, @onboard_timeout_ms) do
      {:ok, result} ->
        identity = %{
          csid: result["client_session_id"],
          agent_uuid: result["uuid"],
          mode: :scratch
        }

        # Harvest the recovery anchor directly from the onboard response —
        # the binding is PG-durable server-side the moment onboard returns
        # (bound_via=onboard_stable_session), and the continuity_token is the
        # client's half of the both-store-loss alarm path.
        anchor = %{
          "agent_uuid" => identity.agent_uuid,
          "client_session_id" => identity.csid,
          "continuity_token" => result["continuity_token"],
          "saved_at" => System.system_time(:second),
          "mode" => "scratch"
        }

        persist_anchor(state.id_file, anchor)
        Logger.info("[Governance.Client] onboarded scratch identity #{identity.agent_uuid}")
        schedule_checkin(5_000)

        {:noreply,
         %{state | identity: identity, anchor: anchor, onboard_retry_ms: @onboard_retry_base_ms}}

      {:error, reason} ->
        Logger.warning(
          "[Governance.Client] onboard failed (#{inspect(reason)}); " <>
            "retrying in #{state.onboard_retry_ms}ms"
        )

        Process.send_after(self(), :onboard, state.onboard_retry_ms)
        next = min(state.onboard_retry_ms * 2, @onboard_retry_cap_ms)
        {:noreply, %{state | onboard_retry_ms: next}}
    end
  end

  def handle_info(:checkin, state) do
    schedule_checkin(state.interval_ms)
    {:noreply, do_checkin(state)}
  end

  defp do_checkin(%{identity: nil} = state), do: state

  defp do_checkin(state) do
    now_ms = System.monotonic_time(:millisecond)

    cond do
      is_integer(state.blocked_until_ms) and now_ms < state.blocked_until_ms ->
        state

      true ->
        case LiveState.read(state.live_state_opts) do
          {:ok, %{"readings" => readings} = live_state} ->
            case Map.get(live_state, "body_anima") || Map.get(live_state, "anima") do
              anima when is_map(anima) ->
                attempt_checkin(state, anima, readings, _may_reanchor? = true)

              _ ->
                Logger.warning("[Governance.Client] live envelope has no body_anima; skipping")

                state
            end

          {:error, reason} ->
            Logger.warning("[Governance.Client] live envelope unavailable (#{reason}); skipping")
            state
        end
    end
  end

  defp attempt_checkin(state, anima, readings, may_reanchor?) do
    body_projection = EisvMapper.anima_to_body_eisv_projection(anima, readings)

    args =
      %{
        "client_session_id" => state.identity.csid,
        "agent_name" => agent_name(state.identity),
        "complexity" => EisvMapper.estimate_complexity(anima, readings),
        "confidence" => EisvMapper.compute_confidence(anima, state.prev_anima),
        "ethical_drift" =>
          EisvMapper.compute_ethical_drift(anima, state.prev_anima, readings, state.prev_readings),
        "response_text" => EisvMapper.status_text(anima, body_projection),
        "sensor_data" => %{
          "body_eisv_projection" => body_projection,
          "body_anima" => anima,
          # Legacy aliases retained until all UNITARES deployments consume
          # the provenance-aware sensor schema.
          "eisv" => body_projection,
          "anima" => anima,
          "state_space_provenance" => %{
            "body_anima" => %{
              "source" => "broker_published_anima",
              "role" => "physical_self_sense"
            },
            "anima" => %{
              "alias_of" => "body_anima",
              "deprecated" => true
            },
            "body_eisv_projection" => %{
              "schema" => "anima.body_eisv_projection.v1",
              "source" => "anima_sensor_projection",
              "role" => "governance_input_measurement"
            },
            "eisv" => %{
              "alias_of" => "body_eisv_projection",
              "deprecated" => true
            }
          },
          "environment" =>
            Map.take(readings, ["ambient_temp_c", "humidity_pct", "light_lux", "pressure_hpa"])
        },
        "response_mode" => "minimal"
        # Deliberately NO "agent_id": the addendum requires presenting the
        # same identity material as the Python bridge (csid echo only). A
        # declared agent_id makes the REST strict gate skip its refusal
        # entirely, so binding loss becomes unobservable — check-ins resolve
        # by uuid passthrough (no PG renewal, no Redis re-cache) and the
        # refusal-detection/anchor loop this client exists to prove never
        # fires. Found live 2026-07-03: the Redis-wipe acceptance test passed
        # WITHOUT touching PATH2 until this key was dropped.
      }

    case call_tool(state, "process_agent_update", args, @checkin_timeout_ms) do
      {:ok, %{"action" => _} = result} ->
        write_governance(result, body_projection, state.identity)

        %{
          state
          | prev_anima: anima,
            prev_readings: readings,
            failures: 0,
            breaker_backoff_s: @breaker_base_s
        }
        |> maybe_refresh_anchor()

      {:ok, other} ->
        # A success-shaped payload WITHOUT an action is not a verdict. The
        # known such shape (typed refusal) is classified in decode_envelope;
        # anything else is treated as a failure — never defaulted to
        # "proceed" (the 2026-06-30 outage mechanism).
        Logger.warning(
          "[Governance.Client] check-in returned no action " <>
            "(keys: #{inspect(Map.keys(other))}); treating as failure"
        )

        trip_breaker(state, {:unexpected_checkin_shape, Map.keys(other)})

      {:error, {:agent_paused, detail}} ->
        # A pause is a governance verdict, not a transport failure: record it
        # in the shadow slice and do NOT open the breaker (Sentinel lesson —
        # swallowing pause as tool_error left a resident dark for hours).
        Logger.warning("[Governance.Client] agent paused: #{inspect(detail)}")

        write_governance(
          %{"action" => "pause", "reason" => "agent_paused"},
          body_projection,
          state.identity
        )

        %{state | prev_anima: anima, prev_readings: readings}

      {:error, {:identity_refused, hint}} ->
        handle_identity_refusal(state, anima, readings, hint, may_reanchor?)

      {:error, reason} ->
        trip_breaker(state, reason)
    end
  end

  defp trip_breaker(state, reason) do
    failures = state.failures + 1

    if failures >= @breaker_threshold do
      backoff_s = state.breaker_backoff_s

      Logger.warning(
        "[Governance.Client] check-in failed (#{inspect(reason)}); " <>
          "breaker open #{backoff_s}s after #{failures} consecutive failures"
      )

      %{
        state
        | failures: failures,
          blocked_until_ms: System.monotonic_time(:millisecond) + backoff_s * 1_000,
          breaker_backoff_s: min(backoff_s * 2, @breaker_cap_s)
      }
    else
      Logger.warning(
        "[Governance.Client] check-in failed (#{inspect(reason)}); #{failures}/#{@breaker_threshold}"
      )

      %{state | failures: failures}
    end
  end

  # -- identity refusal / recovery anchor (#97, addendum amendment) ----------

  defp handle_identity_refusal(state, anima, readings, hint, may_reanchor?) do
    Logger.error(
      "[Governance.Client] IDENTITY REFUSED for #{state.identity.csid}: #{hint} — " <>
        "typed strict refusal (unguarded parses read this as silent \"proceed\"). " <>
        "Governance slice deliberately NOT written; staleness is the alarm channel."
    )

    if may_reanchor? do
      case try_reanchor(state) do
        {:ok, state2} ->
          # Retry ONCE this cycle with the (possibly re-pointed) identity.
          attempt_checkin(state2, anima, readings, false)

        {:skip, state2} ->
          trip_breaker(state2, :identity_refused)
      end
    else
      trip_breaker(state, :identity_refused)
    end
  end

  defp try_reanchor(state) do
    # Re-read the anchor file first: it is the operator lever for re-pointing
    # the echoed key, and may have been rotated since boot.
    anchor = load_anchor(state.id_file) || state.anchor
    now_ms = System.monotonic_time(:millisecond)

    cond do
      anchor == nil or anchor["agent_uuid"] == nil or anchor["continuity_token"] == nil ->
        Logger.error(
          "[Governance.Client] identity refused and NO usable recovery anchor at " <>
            "#{state.id_file || "(no id_file)"} — cannot self-recover. " <>
            "OPERATOR RECOVERY REQUIRED: run #{@operator_runbook} <uuid> #{state.identity.csid}"
        )

        {:skip, %{state | anchor: anchor}}

      is_integer(state.reanchor_last_ms) and
          now_ms - state.reanchor_last_ms < @reanchor_cooldown_ms ->
        {:skip, %{state | anchor: anchor}}

      true ->
        spend_anchor(%{state | anchor: anchor, reanchor_last_ms: now_ms})
    end
  end

  defp spend_anchor(state) do
    anchor = state.anchor

    args = %{
      "agent_uuid" => anchor["agent_uuid"],
      "continuity_token" => anchor["continuity_token"],
      "client_session_id" => state.identity.csid,
      "resume" => true,
      "response_mode" => "minimal"
    }

    case call_tool(state, "identity", args, @identity_timeout_ms) do
      {:ok, resp} ->
        got = resp["uuid"] || get_in(resp, ["bound_identity", "uuid"])

        if got == anchor["agent_uuid"] do
          # Settled ontology (2026-07-03 acceptance tests): identity(resume)
          # RESOLVES this call but never WRITES a binding, and no sanctioned
          # call can (S1-c, S21-a — these gates ARE the phantom-mint fix).
          # A Redis-only wipe never reaches this code (PATH2 self-heals via
          # the PG row); being refused means BOTH stores lost the binding.
          bound_key = resp["client_session_id"] || state.identity.csid

          new_anchor = %{
            anchor
            | "continuity_token" => resp["continuity_token"] || anchor["continuity_token"],
              "client_session_id" => bound_key,
              "saved_at" => System.system_time(:second)
          }

          persist_anchor(state.id_file, new_anchor)

          Logger.error(
            "[Governance.Client] identity VERIFIED (uuid=#{String.slice(got, 0, 8)}) but the " <>
              "session binding is gone from BOTH stores — a client cannot recreate it by " <>
              "design. OPERATOR RECOVERY REQUIRED: run #{@operator_runbook} #{got} #{bound_key}"
          )

          {:ok,
           %{
             state
             | anchor: new_anchor,
               identity: %{state.identity | csid: bound_key, agent_uuid: got}
           }}
        else
          Logger.error(
            "[Governance.Client] re-anchor resolved to #{String.slice(got || "nothing", 0, 8)}, " <>
              "expected #{String.slice(anchor["agent_uuid"], 0, 8)} — refusing mismatched binding"
          )

          {:skip, state}
        end

      {:error, reason} ->
        Logger.error("[Governance.Client] re-anchor attempt failed: #{inspect(reason)}")
        {:skip, state}
    end
  end

  # While the binding is healthy, keep the recovery anchor fresh (daily) so a
  # future store wipe is recoverable/diagnosable. Best-effort — failures only
  # log at debug, exactly like the Python bridge's _harvest_anchor.
  defp maybe_refresh_anchor(state) do
    if anchor_fresh?(state.anchor) do
      state
    else
      args = %{"client_session_id" => state.identity.csid, "response_mode" => "minimal"}

      case call_tool(state, "identity", args, @identity_timeout_ms) do
        {:ok, resp} ->
          uuid = resp["uuid"] || get_in(resp, ["bound_identity", "uuid"])
          token = resp["continuity_token"]

          if uuid && token do
            # Adopt the server-canonical session key (#99): recovery only
            # converges when we echo what the server actually bound.
            bound_key = resp["client_session_id"] || state.identity.csid

            anchor = %{
              "agent_uuid" => uuid,
              "client_session_id" => bound_key,
              "continuity_token" => token,
              "saved_at" => System.system_time(:second),
              "mode" => to_string(state.identity.mode)
            }

            persist_anchor(state.id_file, anchor)

            %{
              state
              | anchor: anchor,
                identity: %{state.identity | csid: bound_key, agent_uuid: uuid}
            }
          else
            Logger.debug("[Governance.Client] anchor harvest: identity() returned no uuid/token")
            state
          end

        {:error, reason} ->
          Logger.debug(
            "[Governance.Client] anchor harvest failed (non-fatal): #{inspect(reason)}"
          )

          state
      end
    end
  end

  defp anchor_fresh?(nil), do: false

  defp anchor_fresh?(anchor) do
    anchor["continuity_token"] != nil and anchor["agent_uuid"] != nil and
      is_number(anchor["saved_at"]) and
      System.system_time(:second) - anchor["saved_at"] < @anchor_refresh_s
  end

  defp write_governance(result, body_projection, identity) do
    {governance_eisv, governance_eisv_source} = extract_governance_eisv(result)

    AnimaBroker.State.Store.merge(%{
      "governance" => %{
        # No "proceed" default — callers guarantee an action is present, and
        # a shape without one must never be written as a verdict (#97).
        "action" => Map.fetch!(result, "action"),
        # No "comfortable" default either: the agent_paused path sends no
        # margin, and a paused Lumen must not be published as comfortable.
        # Python consumers already treat a null margin as unknown.
        "margin" => Map.get(result, "margin"),
        "reason" => Map.get(result, "reason"),
        "body_eisv_projection" => body_projection,
        "governance_eisv" => governance_eisv,
        "governance_eisv_source" => governance_eisv_source,
        # Compatibility only: this has always been the body projection, not
        # UNITARES's primary/behavioral state.
        "eisv" => body_projection,
        "eisv_source" => "body_eisv_projection_legacy_alias",
        "state_space_provenance" => %{
          "body_eisv_projection" => %{
            "schema" => "anima.body_eisv_projection.v1",
            "source" => "anima_sensor_projection",
            "role" => "body_measurement"
          },
          "governance_eisv" => %{
            "source" => governance_eisv_source,
            "role" => "unitares_inferred_state"
          },
          "eisv" => %{
            "alias_of" => "body_eisv_projection",
            "deprecated" => true
          }
        },
        "source" => "unitares_ex",
        "identity_mode" => to_string(identity.mode),
        "unitares_agent_id" => identity.agent_uuid,
        "governance_at" => NaiveDateTime.local_now() |> NaiveDateTime.to_iso8601()
      }
    })
  end

  defp extract_governance_eisv(result) do
    metrics = Map.get(result, "metrics", %{})

    source =
      Map.get(result, "primary_eisv_source") ||
        if(is_map(metrics), do: Map.get(metrics, "primary_eisv_source"), else: nil)

    vector =
      [
        Map.get(result, "primary_eisv"),
        if(is_map(metrics), do: Map.get(metrics, "primary_eisv"), else: nil),
        metrics,
        result,
        Map.get(result, "eisv")
      ]
      |> Enum.find_value(&validated_eisv_vector/1)

    if vector do
      {vector, source || "unitares_primary_response_source_omitted"}
    else
      {nil, "not_returned_by_unitares"}
    end
  end

  defp validated_eisv_vector(candidate) when is_map(candidate) do
    keys = ["E", "I", "S", "V"]

    if Enum.all?(keys, fn key ->
         value = Map.get(candidate, key)
         lower = if key == "V", do: -1.0, else: 0.0
         is_number(value) and value >= lower and value <= 1.0
       end) do
      Map.take(candidate, keys)
    end
  end

  defp validated_eisv_vector(_candidate), do: nil

  # -- REST tool bridge ------------------------------------------------------

  defp call_tool(state, tool, args, timeout_ms) do
    body = %{"name" => tool, "arguments" => args}

    case state.http_post.(state.url, body, headers(), timeout_ms) do
      {:ok, 200, raw} -> decode_envelope(raw)
      {:ok, status, raw} -> {:error, {:http_status, status, String.slice(raw || "", 0, 200)}}
      {:error, reason} -> {:error, {:transport, reason}}
    end
  end

  defp decode_envelope(raw) do
    case Jason.decode(raw) do
      {:ok, %{"success" => true, "result" => result}} ->
        result = normalize_result(result)

        case refusal_hint(result) do
          nil -> {:ok, result}
          hint -> {:error, {:identity_refused, hint}}
        end

      {:ok, %{"success" => false, "error_code" => "AGENT_PAUSED"} = err} ->
        {:error, {:agent_paused, err["error"] || err["detail"]}}

      {:ok, %{"success" => false} = err} ->
        case refusal_hint(err) do
          nil -> {:error, {:tool_error, err["error_code"] || err["error"]}}
          hint -> {:error, {:identity_refused, hint}}
        end

      {:ok, other} ->
        {:error, {:unexpected_envelope, other}}

      {:error, _} ->
        {:error, :bad_json}
    end
  end

  # The #425 typed strict refusal is a structured SUCCESS-shape (no
  # success:false, no action) — single-sourced server-side in
  # strict_identity_refusal_payload and returned identically by the REST
  # gate, the MCP dispatch middleware, and the process_agent_update Path-C
  # refusal. Without this classification it parses as a silent
  # default-"proceed" (the 2026-06-30→07-02 outage mechanism, anima-mcp #97).
  defp refusal_hint(%{"status" => status} = r)
       when status in ["identity_required", "lineage_declaration_required"],
       do: r["hint"] || r["error"] || "session binding unresolved (status=#{status})"

  defp refusal_hint(%{"error_code" => "SESSION_ERROR"} = r),
    do: r["hint"] || r["error"] || "SESSION_ERROR"

  defp refusal_hint(%{"error_category" => "auth_error"} = r),
    do: r["hint"] || r["error"] || "auth_error"

  defp refusal_hint(_), do: nil

  # The bridge sometimes returns the tool result as a JSON string rather than
  # an object (dispatch_beam handles the same).
  defp normalize_result(result) when is_map(result), do: result

  defp normalize_result(result) when is_binary(result) do
    case Jason.decode(result) do
      {:ok, %{} = m} -> m
      _ -> %{"raw" => result}
    end
  end

  defp normalize_result(other), do: %{"raw" => other}

  defp headers do
    base = [{~c"content-type", ~c"application/json"}]

    case System.get_env("UNITARES_HTTP_API_TOKEN") do
      nil -> base
      "" -> base
      token -> [{~c"authorization", String.to_charlist("Bearer " <> token)} | base]
    end
  end

  defp httpc_post(url, body_map, headers, timeout_ms) do
    request = {String.to_charlist(url), headers, ~c"application/json", Jason.encode!(body_map)}

    case :httpc.request(:post, request, [timeout: timeout_ms], body_format: :binary) do
      {:ok, {{_, status, _}, _resp_headers, body}} -> {:ok, status, body}
      {:error, reason} -> {:error, reason}
    end
  end

  # -- anchor persistence (parent chaining + recovery, one file) --------------

  defp load_prior_uuid(path) do
    case load_anchor(path) do
      %{"agent_uuid" => uuid} when is_binary(uuid) -> uuid
      _ -> nil
    end
  end

  defp load_anchor(nil), do: nil

  defp load_anchor(path) do
    with {:ok, raw} <- File.read(Path.expand(path)),
         {:ok, %{} = data} <- Jason.decode(raw) do
      # Accept the Python anchor's "uuid" key too — the file is an operator
      # lever and a hand-edit must not fail on naming.
      Map.put_new(data, "agent_uuid", data["uuid"])
    else
      _ -> nil
    end
  end

  defp persist_anchor(nil, _anchor), do: :ok

  defp persist_anchor(path, anchor) do
    path = Path.expand(path)
    File.mkdir_p!(Path.dirname(path))
    tmp = path <> ".tmp"
    File.write!(tmp, Jason.encode!(anchor))
    File.rename!(tmp, path)
  rescue
    e -> Logger.warning("[Governance.Client] could not persist anchor: #{inspect(e)}")
  end

  defp agent_name(%{mode: :substrate}), do: "Lumen"
  defp agent_name(_), do: "lumen-broker-ex"

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  defp schedule_checkin(ms), do: Process.send_after(self(), :checkin, ms)
end
