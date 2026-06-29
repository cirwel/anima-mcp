import Config

# Read deployment overrides from the environment at release start, matching how
# the Python services are configured. Skipped in :test (see config.exs).
if config_env() != :test do
  config :anima_broker,
    shm_path: System.get_env("ANIMA_SHM_PATH") || "/dev/shm/anima_state.shadow.json",
    tick_interval_ms: String.to_integer(System.get_env("ANIMA_TICK_MS") || "2000")
end
