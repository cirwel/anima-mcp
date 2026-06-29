defmodule AnimaBroker.MixProject do
  use Mix.Project

  def project do
    [
      app: :anima_broker,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # OTP application. AnimaBroker.Application boots the supervision tree.
  def application do
    [
      extra_applications: [:logger],
      mod: {AnimaBroker.Application, []}
    ]
  end

  defp deps do
    [
      {:jason, "~> 1.4"}
      # Phase 1 adds: {:circuits_i2c, "~> 2.0"}
    ]
  end
end
