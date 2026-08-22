defmodule AnimaBroker.Governance.ClientTest do
  # Not async: exercises the shared State.Store.
  use ExUnit.Case, async: false

  alias AnimaBroker.Governance.Client
  alias AnimaBroker.State.Store

  @onboard_result %{
    "uuid" => "test-uuid-1234",
    "client_session_id" => "agent-test-1234",
    "agent_id" => "TestAgent",
    "continuity_token" => "v1.token-from-onboard"
  }

  # The #425 typed strict refusal as the REST gate actually returns it:
  # success:true envelope, refusal payload as the result — NO success:false,
  # NO action. The exact shape that parsed as silent "proceed" (#97).
  @typed_refusal %{
    "status" => "identity_required",
    "tool" => "process_agent_update",
    "hint" => "session binding unresolved",
    "next_step" => "onboard"
  }

  # Every file this suite writes goes in a directory unique to THIS test, in
  # THIS VM, and is deleted afterwards.
  #
  # The flake in #167 was not process leakage. Paths were built as
  # `tmp_path("gov_id_#{System.unique_integer([:positive])}.json")`,
  # and `unique_integer` is unique within a VM but restarts in the same numeric
  # range on every `mix test`, while the tmp dir persists. So a client's
  # `load_anchor/1` at init would read an anchor left by an EARLIER RUN of the
  # suite and start up wearing a previous test's identity — which is exactly
  # what the two canonical failures showed (`test-uuid-1234` where the test
  # never onboards, `anchored-uuid-1234` from `write_anchor!/2`'s default).
  #
  # Measured on this branch, same seeds: cleared tmp 0/10 runs failed;
  # accumulating tmp 1/10; tmp holding 400 stale anchors 9/10. That dose
  # response is the whole flake, and it explains the reported oddities — CI
  # looked version-specific because each runner starts clean and only the one
  # that happened to collide failed; running the file alone passed because it
  # creates fewer clients; and the "residual non-determinism at fixed seed" was
  # just tmp contents differing between runs.
  setup do
    dir =
      Path.join(
        System.tmp_dir!(),
        "anima_client_test_#{System.pid()}_#{System.unique_integer([:positive])}"
      )

    File.rm_rf!(dir)
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)
    Process.put(:test_tmp_dir, dir)
    :ok
  end

  defp tmp_path(name), do: Path.join(Process.get(:test_tmp_dir), name)

  defp fresh_live_envelope!(name) do
    path = tmp_path(name)

    File.write!(
      path,
      Jason.encode!(%{
        "updated_at" => NaiveDateTime.to_iso8601(NaiveDateTime.local_now()),
        "pid" => 1,
        "data" => %{
          "anima" => %{
            "warmth" => 0.4,
            "clarity" => 0.8,
            "stability" => 0.7,
            "presence" => 0.6
          },
          "readings" => %{"ambient_temp_c" => 24.5, "cpu_percent" => 10.0}
        }
      })
    )

    path
  end

  defp start_client(http_post, extra_opts \\ []) do
    live = fresh_live_envelope!("client_test_live_#{System.unique_integer([:positive])}.json")

    id_file =
      Keyword.get(
        extra_opts,
        :id_file,
        tmp_path("gov_id_#{System.unique_integer([:positive])}.json")
      )

    opts =
      Keyword.merge(
        [
          name: nil,
          url: "http://localhost:9/v1/tools/call",
          interval_ms: 3_600_000,
          http_post: http_post,
          id_file: id_file,
          live_state_opts: [path: live]
        ],
        extra_opts
      )

    {:ok, pid} = GenServer.start_link(Client, opts)
    %{pid: pid, id_file: id_file, live: live}
  end

  defp ok_envelope(result),
    do: {:ok, 200, Jason.encode!(%{"success" => true, "result" => result})}

  defp write_anchor!(id_file, overrides) do
    anchor =
      Map.merge(
        %{
          "agent_uuid" => "anchored-uuid-1234",
          "client_session_id" => "agent-anchored-123",
          "continuity_token" => "v1.anchored-token",
          "saved_at" => System.system_time(:second),
          "mode" => "scratch"
        },
        overrides
      )

    File.write!(id_file, Jason.encode!(anchor))
    anchor
  end

  test "onboards, echoes csid on check-in, writes governance slice to the Store" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          ok_envelope(%{"action" => "proceed", "margin" => "comfortable", "reason" => "ok"})

        "identity" ->
          ok_envelope(@onboard_result)
      end
    end

    %{pid: pid, id_file: id_file} = start_client(http_post)

    assert_receive {:post, %{"name" => "onboard", "arguments" => onboard_args}}, 2_000
    assert onboard_args["force_new"] == true
    assert onboard_args["name"] == "lumen-broker-ex-shadow"

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => args}}, 2_000

    # Strict identity: the onboard-echoed CSID must ride in arguments.
    assert args["client_session_id"] == "agent-test-1234"
    assert args["agent_name"] == "lumen-broker-ex"
    assert args["response_mode"] == "minimal"
    assert is_number(args["complexity"]) and is_number(args["confidence"])
    assert [_, _, _] = args["ethical_drift"]
    assert %{"eisv" => %{"E" => _}} = args["sensor_data"]

    # Governance decision landed in the (shadow) Store slice.
    wait_until(fn -> get_in(Store.snapshot(), ["governance", "action"]) == "proceed" end)
    gov = Store.snapshot()["governance"]
    assert gov["source"] == "unitares_ex"
    assert gov["identity_mode"] == "scratch"
    assert gov["unitares_agent_id"] == "test-uuid-1234"
    assert is_binary(gov["governance_at"])

    # Recovery anchor persisted (parent chaining + both-store-loss rescue):
    # the continuity_token is harvested straight from the onboard response.
    assert %{
             "agent_uuid" => "test-uuid-1234",
             "continuity_token" => "v1.token-from-onboard",
             "mode" => "scratch"
           } = Jason.decode!(File.read!(id_file))
  end

  test "prior uuid is declared as parent_agent_id on re-onboard" do
    me = self()
    id_file = tmp_path("gov_id_#{System.unique_integer([:positive])}.json")
    File.write!(id_file, Jason.encode!(%{"agent_uuid" => "prior-uuid-999"}))

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})
      ok_envelope(@onboard_result)
    end

    live = fresh_live_envelope!("client_test_live_#{System.unique_integer([:positive])}.json")

    {:ok, _pid} =
      GenServer.start_link(Client,
        name: nil,
        url: "http://localhost:9/v1/tools/call",
        interval_ms: 3_600_000,
        http_post: http_post,
        id_file: id_file,
        live_state_opts: [path: live]
      )

    assert_receive {:post, %{"name" => "onboard", "arguments" => args}}, 2_000
    assert args["parent_agent_id"] == "prior-uuid-999"
  end

  test "breaker opens after 2 consecutive failures and blocks the next attempt" do
    me = self()
    {:ok, agent} = Agent.start_link(fn -> 0 end)

    http_post = fn _url, body, _headers, _timeout ->
      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          Agent.update(agent, &(&1 + 1))
          send(me, {:checkin_attempt, Agent.get(agent, & &1)})
          {:error, :econnrefused}
      end
    end

    %{pid: pid} = start_client(http_post)
    # wait for onboard to complete
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive {:checkin_attempt, 1}, 2_000
    send(pid, :checkin)
    assert_receive {:checkin_attempt, 2}, 2_000

    # Breaker now open (2 consecutive failures): the third tick is skipped.
    send(pid, :checkin)
    refute_receive {:checkin_attempt, 3}, 300

    state = :sys.get_state(pid)
    assert state.failures == 2
    assert state.blocked_until_ms > System.monotonic_time(:millisecond)
  end

  test "AGENT_PAUSED is recorded as a pause decision, not a breaker trip" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          send(me, :checkin_attempt)

          {:ok, 200,
           Jason.encode!(%{
             "success" => false,
             "error_code" => "AGENT_PAUSED",
             "error" => "paused pending review"
           })}

        "identity" ->
          ok_envelope(@onboard_result)
      end
    end

    %{pid: pid} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive :checkin_attempt, 2_000

    wait_until(fn -> get_in(Store.snapshot(), ["governance", "action"]) == "pause" end)
    assert :sys.get_state(pid).failures == 0
  end

  test "fixed csid mode skips onboard and claims the substrate identity" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "process_agent_update" -> ok_envelope(%{"action" => "proceed"})
        "identity" -> ok_envelope(%{})
      end
    end

    %{pid: pid} = start_client(http_post, fixed_csid: "lumen-substrate-csid")

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => args}}, 2_000
    assert args["client_session_id"] == "lumen-substrate-csid"
    assert args["agent_name"] == "Lumen"
    refute_received {:post, %{"name" => "onboard"}}
  end

  test "does not start when no url is configured" do
    assert :ignore = GenServer.start_link(Client, name: nil, url: nil)
  end

  # -- refusal detection + recovery anchor (#97 amendment) -------------------

  test "typed refusal is NEVER written to the governance slice as proceed" do
    me = self()
    Store.merge(%{"governance" => %{"action" => "sentinel-prior"}})

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "onboard" -> ok_envelope(@onboard_result)
        # Refusal envelope exactly as the REST gate returns it: success:true.
        "process_agent_update" -> ok_envelope(@typed_refusal)
        # Spend path: refuse the resume too (both stores gone, token stale).
        "identity" -> ok_envelope(@typed_refusal)
      end
    end

    %{pid: pid} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update"}}, 2_000
    # The anchor exists (harvested at onboard), so a spend attempt follows.
    assert_receive {:post, %{"name" => "identity", "arguments" => id_args}}, 2_000
    assert id_args["resume"] == true
    assert id_args["agent_uuid"] == "test-uuid-1234"
    assert id_args["continuity_token"] == "v1.token-from-onboard"

    # The refusal must surface as a failure, and the slice must keep the
    # prior verdict — a refusal is not a "proceed".
    wait_until(fn -> :sys.get_state(pid).failures > 0 end)
    assert get_in(Store.snapshot(), ["governance", "action"]) == "sentinel-prior"
  end

  test "refusal with a valid anchor: spend verifies uuid, adopts canonical key, retries once" do
    me = self()
    {:ok, checkins} = Agent.start_link(fn -> 0 end)

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          n = Agent.get_and_update(checkins, &{&1 + 1, &1 + 1})

          if n == 1 do
            ok_envelope(@typed_refusal)
          else
            ok_envelope(%{"action" => "proceed", "margin" => "comfortable"})
          end

        "identity" ->
          # resume=true verification: same uuid, server-canonical key + fresh token.
          ok_envelope(%{
            "uuid" => "test-uuid-1234",
            "client_session_id" => "agent-canonical-999",
            "continuity_token" => "v1.rotated-token"
          })
      end
    end

    %{pid: pid, id_file: id_file} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => a1}}, 2_000
    assert a1["client_session_id"] == "agent-test-1234"
    assert_receive {:post, %{"name" => "identity"}}, 2_000

    # Retry rides the ADOPTED server-canonical key (#99), not the old one.
    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => a2}}, 2_000
    assert a2["client_session_id"] == "agent-canonical-999"

    wait_until(fn -> get_in(Store.snapshot(), ["governance", "action"]) == "proceed" end)
    assert :sys.get_state(pid).failures == 0

    # Anchor rotated on disk: canonical key + fresh token survive a restart.
    anchor = Jason.decode!(File.read!(id_file))
    assert anchor["client_session_id"] == "agent-canonical-999"
    assert anchor["continuity_token"] == "v1.rotated-token"
  end

  test "refusal with NO anchor: no spend attempt, counted as failure" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})
      ok_envelope(@typed_refusal)
    end

    # Substrate mode with no anchor file — nothing to spend.
    %{pid: pid} = start_client(http_post, fixed_csid: "lumen-substrate-csid")

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update"}}, 2_000
    refute_receive {:post, %{"name" => "identity"}}, 300

    wait_until(fn -> :sys.get_state(pid).failures > 0 end)
  end

  test "re-anchor refuses a uuid mismatch and does not adopt the binding" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          ok_envelope(@typed_refusal)

        "identity" ->
          ok_envelope(%{
            "uuid" => "SOMEONE-ELSES-UUID",
            "client_session_id" => "agent-hijack-000",
            "continuity_token" => "v1.wrong"
          })
      end
    end

    %{pid: pid} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "identity"}}, 2_000

    # No retry (mismatch = no rebind), identity unchanged, failure counted.
    wait_until(fn -> :sys.get_state(pid).failures > 0 end)
    state = :sys.get_state(pid)
    assert state.identity.csid == "agent-test-1234"
    assert state.identity.agent_uuid == "test-uuid-1234"
  end

  test "re-anchor attempts are rate-limited by the cooldown" do
    me = self()
    {:ok, id_calls} = Agent.start_link(fn -> 0 end)

    http_post = fn _url, body, _headers, _timeout ->
      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          send(me, :checkin_attempt)
          ok_envelope(@typed_refusal)

        "identity" ->
          Agent.update(id_calls, &(&1 + 1))
          send(me, :identity_attempt)
          ok_envelope(@typed_refusal)
      end
    end

    %{pid: pid} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive :checkin_attempt, 2_000
    assert_receive :identity_attempt, 2_000

    # Second refused check-in inside the cooldown: no second spend. Clear the
    # breaker window so the refusal itself (not the breaker) is what gates.
    :sys.replace_state(pid, fn s -> %{s | blocked_until_ms: nil, failures: 0} end)
    send(pid, :checkin)
    assert_receive :checkin_attempt, 2_000
    refute_receive :identity_attempt, 300
    assert Agent.get(id_calls, & &1) == 1
  end

  test "healthy check-in refreshes a missing anchor via identity() and adopts the canonical key" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "process_agent_update" ->
          ok_envelope(%{"action" => "proceed"})

        "identity" ->
          ok_envelope(%{
            "uuid" => "lumen-real-uuid-2522",
            "client_session_id" => "agent-lumen-canon",
            "continuity_token" => "v1.harvested-token"
          })
      end
    end

    # Substrate mode, no anchor yet: first healthy check-in must harvest one.
    %{pid: pid, id_file: id_file} =
      start_client(http_post, fixed_csid: "lumen-substrate-csid")

    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => a1}}, 2_000
    assert a1["client_session_id"] == "lumen-substrate-csid"
    assert_receive {:post, %{"name" => "identity", "arguments" => id_args}}, 2_000
    refute Map.has_key?(id_args, "resume")

    wait_until(fn -> File.exists?(id_file) end)
    anchor = Jason.decode!(File.read!(id_file))
    assert anchor["agent_uuid"] == "lumen-real-uuid-2522"
    assert anchor["continuity_token"] == "v1.harvested-token"
    assert anchor["mode"] == "substrate"

    # Canonical-key adoption (#99): the next check-in echoes what the server
    # actually bound, not the bootstrap literal.
    send(pid, :checkin)

    assert_receive {:post,
                    %{
                      "name" => "process_agent_update",
                      "arguments" => %{"client_session_id" => "agent-lumen-canon"}
                    }},
                   2_000
  end

  test "fresh anchor is not re-harvested on every healthy check-in" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "onboard" -> ok_envelope(@onboard_result)
        "process_agent_update" -> ok_envelope(%{"action" => "proceed"})
        "identity" -> ok_envelope(@onboard_result)
      end
    end

    %{pid: pid} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    # Anchor was harvested at onboard (fresh, with token): no identity call.
    send(pid, :checkin)
    assert_receive {:post, %{"name" => "process_agent_update"}}, 2_000
    refute_receive {:post, %{"name" => "identity"}}, 300
  end

  test "substrate mode prefers a substrate anchor's canonical key over the env literal" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})
      ok_envelope(%{"action" => "proceed"})
    end

    id_file = tmp_path("gov_id_#{System.unique_integer([:positive])}.json")

    write_anchor!(id_file, %{
      "mode" => "substrate",
      "client_session_id" => "agent-lumen-canon",
      "agent_uuid" => "lumen-real-uuid-2522"
    })

    %{pid: pid} =
      start_client(http_post, fixed_csid: "lumen-substrate-csid", id_file: id_file)

    send(pid, :checkin)

    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => args}}, 2_000
    assert args["client_session_id"] == "agent-lumen-canon"
    # Echo-only identity material: no agent_id even when the uuid is known —
    # a declared agent_id would make the strict gate skip refusals entirely.
    refute Map.has_key?(args, "agent_id")
  end

  test "substrate mode ignores a scratch anchor (soak identity must not shadow the operator key)" do
    me = self()

    http_post = fn _url, body, _headers, _timeout ->
      send(me, {:post, body})

      case body["name"] do
        "process_agent_update" -> ok_envelope(%{"action" => "proceed"})
        "identity" -> ok_envelope(%{})
      end
    end

    id_file = tmp_path("gov_id_#{System.unique_integer([:positive])}.json")
    write_anchor!(id_file, %{"mode" => "scratch", "client_session_id" => "agent-soak-scratch"})

    %{pid: pid} =
      start_client(http_post, fixed_csid: "lumen-substrate-csid", id_file: id_file)

    send(pid, :checkin)

    assert_receive {:post, %{"name" => "process_agent_update", "arguments" => args}}, 2_000
    assert args["client_session_id"] == "lumen-substrate-csid"
  end

  test "success-shaped result without an action is a failure, never a proceed" do
    me = self()
    Store.merge(%{"governance" => %{"action" => "sentinel-prior-2"}})

    http_post = fn _url, body, _headers, _timeout ->
      case body["name"] do
        "onboard" ->
          ok_envelope(@onboard_result)

        "process_agent_update" ->
          send(me, :checkin_attempt)
          # Not a refusal, not a verdict — e.g. a future server shape drift.
          ok_envelope(%{"acknowledged" => true})
      end
    end

    %{pid: pid} = start_client(http_post)
    wait_until(fn -> :sys.get_state(pid).identity != nil end)

    send(pid, :checkin)
    assert_receive :checkin_attempt, 2_000

    wait_until(fn -> :sys.get_state(pid).failures > 0 end)
    assert get_in(Store.snapshot(), ["governance", "action"]) == "sentinel-prior-2"
  end

  defp wait_until(fun, tries \\ 40) do
    cond do
      fun.() ->
        :ok

      tries == 0 ->
        flunk("condition never became true")

      true ->
        Process.sleep(50)
        wait_until(fun, tries - 1)
    end
  end
end
