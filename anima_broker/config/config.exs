import Config

# Phase 0 writes to a SHADOW path, NOT the live SHM file the Python server
# reads. This lets the Elixir broker run alongside the Python broker with zero
# risk to the live creature. Cutover to the live `/dev/shm/anima_state.json`
# happens in Phase 1, once sensor readings are real and validated.
config :anima_broker,
  shm_path: "/dev/shm/anima_state.shadow.json",
  tick_interval_ms: 2000,
  autostart: true

# In tests we don't want the tick loop spinning and writing to /dev/shm.
if config_env() == :test do
  config :anima_broker, autostart: false
end
