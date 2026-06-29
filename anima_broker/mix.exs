defmodule AnimaBroker.MixProject do
  use Mix.Project

  def project do
    [
      app: :anima_broker,
      version: "0.1.0",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # OTP application. AnimaBroker.Application boots the supervision tree.
  def application do
    [
      extra_applications: [:logger],
      mod: {AnimaBroker.Application, []}
    ]
  end

  defp deps do
    [
      {:jason, "~> 1.4"},
      # I2C sensor bus. On non-Linux hosts (e.g. macOS dev) circuits_i2c
      # builds against a stub backend, so it compiles without hardware; the
      # real NIF is used on the Pi. Drivers go through AnimaBroker.Hardware.I2C
      # so tests inject a fake bus and never touch this dep.
      {:circuits_i2c, "~> 2.0"}
    ]
  end
end
