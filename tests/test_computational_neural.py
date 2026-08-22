"""
Tests for computational neural sensor module.

Validates neural band derivation from system metrics.
"""

import pytest
import time
from unittest.mock import patch
from collections import namedtuple

from anima_mcp.computational_neural import (
    ComputationalNeuralSensor,
)


# Mock types matching psutil's named tuples
CpuStats = namedtuple("scpustats", ["ctx_switches", "interrupts", "soft_interrupts", "syscalls"])
DiskIO = namedtuple("sdiskio", ["read_count", "write_count", "read_bytes", "write_bytes", "read_time", "write_time"])
CpuTimes = namedtuple("scputimes", ["user", "nice", "system", "idle", "iowait"])
NetIO = namedtuple("snetio", ["bytes_sent", "bytes_recv", "packets_sent", "packets_recv", "errin", "errout", "dropin", "dropout"])


def _mock_psutil(mock_ps, cpu_stats=None, disk_io=None, iowait=0.0):
    """Configure psutil mock with proper return values."""
    if cpu_stats is None:
        mock_ps.cpu_stats.return_value = CpuStats(
            ctx_switches=100000, interrupts=50000, soft_interrupts=0, syscalls=0
        )
    else:
        mock_ps.cpu_stats.return_value = cpu_stats
    if disk_io is None:
        mock_ps.disk_io_counters.return_value = DiskIO(
            read_count=0, write_count=0, read_bytes=0, write_bytes=0, read_time=0, write_time=0
        )
    else:
        mock_ps.disk_io_counters.return_value = disk_io
    # Mock cpu_times_percent for theta/iowait calculation
    mock_ps.cpu_times_percent.return_value = CpuTimes(
        user=20.0, nice=0.0, system=10.0, idle=70.0 - iowait, iowait=iowait
    )
    # Mock net_io_counters for theta network I/O
    mock_ps.net_io_counters.return_value = NetIO(
        bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0,
        errin=0, errout=0, dropin=0, dropout=0
    )


@pytest.fixture
def sensor():
    """Create a fresh sensor with no history."""
    return ComputationalNeuralSensor(window_size=10)


class TestBetaBand:
    """Test Beta band: CPU percent → active processing."""

    def test_zero_cpu_gives_zero_beta(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=0.0, memory_percent=50.0)
        assert state.beta == 0.0

    def test_full_cpu_gives_max_beta(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=100.0, memory_percent=50.0)
        assert state.beta == 1.0

    def test_half_cpu_gives_half_beta(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
        assert state.beta == 0.5


class TestAlphaBand:
    """Test Alpha band: CPU idle fraction (1 - beta)."""

    def test_idle_cpu_gives_high_alpha(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=0.0, memory_percent=50.0)
        assert state.alpha == 1.0

    def test_full_cpu_gives_zero_alpha(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=100.0, memory_percent=50.0)
        assert state.alpha == 0.0

    def test_alpha_is_inverse_beta(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=30.0, memory_percent=50.0)
        assert state.alpha == pytest.approx(1.0 - state.beta)

    def test_alpha_independent_of_memory(self, sensor):
        """Alpha no longer uses memory — same CPU should give same alpha."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state_low_mem = sensor.get_neural_state(cpu_percent=50.0, memory_percent=20.0)
        sensor2 = ComputationalNeuralSensor()
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state_high_mem = sensor2.get_neural_state(cpu_percent=50.0, memory_percent=90.0)
        assert state_low_mem.alpha == state_high_mem.alpha


class TestGammaBand:
    """Test Gamma band: context switches + interrupts → spiking activity."""

    def test_first_sample_zero_gamma(self, sensor):
        """First sample has no previous stats → gamma = 0."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
        assert state.gamma == 0.0

    def test_high_ctx_switches_high_gamma(self, sensor):
        """High context switch rate → high gamma (with EMA smoothing)."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            # First sample: baseline (gamma=0, EMA initialized to 0)
            _mock_psutil(mock_ps, cpu_stats=CpuStats(100000, 50000, 0, 0))
            sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

            # Sustained high ctx switches over several samples to let EMA converge
            for i in range(5):
                sensor._last_sample_time = time.monotonic() - 2.0
                _mock_psutil(mock_ps, cpu_stats=CpuStats(100000 + 40000 * (i + 1), 50000, 0, 0))
                state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

        # Raw gamma = 0.6 each step. After 5 EMA steps (alpha=0.2):
        # converges toward 0.6, should be above 0.3
        assert state.gamma > 0.3
        assert state.gamma < 0.6  # EMA hasn't fully converged

    def test_gamma_ema_uses_elapsed_wall_time(self):
        short = ComputationalNeuralSensor()
        long = ComputationalNeuralSensor()
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps, cpu_stats=CpuStats(100000, 50000, 0, 0))
            short.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
            short._last_sample_time = time.monotonic() - 2.0
            _mock_psutil(mock_ps, cpu_stats=CpuStats(110000, 50000, 0, 0))
            short_state = short.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps, cpu_stats=CpuStats(100000, 50000, 0, 0))
            long.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
            long._last_sample_time = time.monotonic() - 10.0
            _mock_psutil(mock_ps, cpu_stats=CpuStats(150000, 50000, 0, 0))
            long_state = long.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

        # Both windows have the same raw switching rate; the longer elapsed
        # interval must advance the continuous-time EMA farther.
        assert long_state.gamma > short_state.gamma * 2

    def test_gamma_independent_of_cpu(self, sensor):
        """Gamma should NOT track CPU percent — it tracks switching rate."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            # First sample
            _mock_psutil(mock_ps, cpu_stats=CpuStats(100000, 50000, 0, 0))
            sensor.get_neural_state(cpu_percent=10.0, memory_percent=50.0)

            # Multiple samples with same switching rate to let EMA converge
            for i in range(3):
                sensor._last_sample_time = time.monotonic() - 1.0
                _mock_psutil(mock_ps, cpu_stats=CpuStats(100000 + 5000 * (i + 1), 50000 + 5000 * (i + 1), 0, 0))
                state_low_cpu = sensor.get_neural_state(cpu_percent=10.0, memory_percent=50.0)

        sensor2 = ComputationalNeuralSensor()
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps, cpu_stats=CpuStats(100000, 50000, 0, 0))
            sensor2.get_neural_state(cpu_percent=90.0, memory_percent=50.0)

            for i in range(3):
                sensor2._last_sample_time = time.monotonic() - 1.0
                _mock_psutil(mock_ps, cpu_stats=CpuStats(100000 + 5000 * (i + 1), 50000 + 5000 * (i + 1), 0, 0))
                state_high_cpu = sensor2.get_neural_state(cpu_percent=90.0, memory_percent=50.0)

        # Same switching rate → same gamma, regardless of CPU
        assert state_low_cpu.gamma == pytest.approx(state_high_cpu.gamma, abs=0.01)
        # But beta should differ
        assert state_low_cpu.beta != state_high_cpu.beta


class TestThetaBand:
    """Test Theta band: disk busy_time + network I/O → integration."""

    def test_no_io_zero_theta(self, sensor):
        """No disk or network I/O → theta = 0."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            # Prime with first sample
            sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
            sensor._last_sample_time = time.monotonic() - 1.0
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
        assert state.theta == 0.0

    def test_disk_busy_raises_theta(self, sensor):
        """Disk busy_time ratio should raise theta (with doubled ceiling and EMA)."""
        DiskIOBusy = namedtuple("sdiskio", ["read_count", "write_count", "read_bytes", "write_bytes", "read_time", "write_time", "busy_time"])
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            # First sample: baseline disk counters
            disk1 = DiskIOBusy(0, 0, 0, 0, 0, 0, busy_time=0)
            _mock_psutil(mock_ps, disk_io=disk1)
            sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

            # Second sample: 500ms of busy_time in 1 second
            # raw disk_signal = 500 / (1.0 * 2000) = 0.25 (doubled ceiling)
            # theta blend = 0.7 * 0.25 + 0.3 * 0 = 0.175 (net=0)
            # EMA first real value: initialized to 0 on first call, so
            # ema = 0.3 * 0.175 + 0.7 * 0.0 = 0.0525
            # Need a second identical sample to converge closer
            sensor._last_sample_time = time.monotonic() - 1.0
            disk2 = DiskIOBusy(0, 0, 0, 0, 0, 0, busy_time=500)
            _mock_psutil(mock_ps, disk_io=disk2)
            sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

            # Third sample: same delta to let EMA converge
            sensor._last_sample_time = time.monotonic() - 1.0
            disk3 = DiskIOBusy(0, 0, 0, 0, 0, 0, busy_time=1000)
            _mock_psutil(mock_ps, disk_io=disk3)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

        # After 2 EMA steps the value should be meaningfully above zero
        assert state.theta > 0.05
        assert state.theta < 0.25  # well below old 0.5 — ceiling + EMA dampen it

    def test_network_io_raises_theta(self, sensor):
        """Network throughput should raise theta (with weighted blend and EMA)."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            # First sample: baseline net counters
            _mock_psutil(mock_ps)
            mock_ps.net_io_counters.return_value = NetIO(0, 0, 0, 0, 0, 0, 0, 0)
            sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

            # Second sample: 250KB in 1 second
            # net_signal = 256KB / 512KB = 0.5
            # theta blend = 0.7*0.5 + 0.3*0 = 0.35 (disk=0)
            # EMA: 0.3*0.35 + 0.7*0 = 0.105
            sensor._last_sample_time = time.monotonic() - 1.0
            _mock_psutil(mock_ps)
            mock_ps.net_io_counters.return_value = NetIO(128*1024, 128*1024, 0, 0, 0, 0, 0, 0)
            sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

            # Third sample: same throughput to let EMA converge
            sensor._last_sample_time = time.monotonic() - 1.0
            _mock_psutil(mock_ps)
            mock_ps.net_io_counters.return_value = NetIO(256*1024, 256*1024, 0, 0, 0, 0, 0, 0)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

        assert state.theta > 0.1
        assert state.theta < 0.4  # EMA dampens below raw 0.5

    def test_ema_dampens_spike(self, sensor):
        """A single I/O spike should be dampened by EMA smoothing."""
        DiskIOBusy = namedtuple("sdiskio", ["read_count", "write_count", "read_bytes", "write_bytes", "read_time", "write_time", "busy_time"])
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            # Baseline: zero I/O for several samples to set EMA near 0
            for i in range(3):
                disk = DiskIOBusy(0, 0, 0, 0, 0, 0, busy_time=0)
                _mock_psutil(mock_ps, disk_io=disk)
                if i > 0:
                    sensor._last_sample_time = time.monotonic() - 1.0
                sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

            # Now spike: 2000ms busy in 1 second = raw disk_signal 1.0
            sensor._last_sample_time = time.monotonic() - 1.0
            disk_spike = DiskIOBusy(0, 0, 0, 0, 0, 0, busy_time=2000)
            _mock_psutil(mock_ps, disk_io=disk_spike)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)

        # Raw theta would be 0.7*1.0 + 0.3*0 = 0.7, but EMA dampens heavily
        # EMA was near 0, so: 0.3*0.7 + 0.7*~0 ≈ 0.21
        assert state.theta < 0.35  # well below raw 0.7
        assert state.theta > 0.0   # but not zero


class TestDeltaBand:
    """Test Delta band: CPU variance stability + temp stability."""

    def test_steady_cpu_high_delta(self, sensor):
        """Steady CPU (even high) = stable = high delta."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            # Feed several samples at steady 90% CPU
            for _ in range(5):
                sensor.get_neural_state(cpu_percent=90.0, memory_percent=50.0)
            state = sensor.get_neural_state(cpu_percent=90.0, memory_percent=50.0)
        # All samples at 90% → range=0 → cpu_stability=1.0
        assert state.delta == pytest.approx(1.0, abs=0.01)

    def test_jumping_cpu_low_delta(self, sensor):
        """Jumping between 10% and 90% CPU = unstable = low delta."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            for cpu in [10.0, 90.0, 10.0, 90.0, 10.0]:
                sensor.get_neural_state(cpu_percent=cpu, memory_percent=50.0)
            state = sensor.get_neural_state(cpu_percent=90.0, memory_percent=50.0)
        # Range = 80 → cpu_stability = max(0, 1 - 80/40) = 0.0
        assert state.delta == pytest.approx(0.3, abs=0.01)  # 0.0 * 0.7 + 1.0 * 0.3

    def test_single_sample_full_delta(self, sensor):
        """First sample with no history → stable by default."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
        # Only 1 sample → cpu_stability=1.0
        assert state.delta == 1.0

    def test_moderate_swing_moderate_delta(self, sensor):
        """20-point CPU swing → partial stability."""
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            for cpu in [40.0, 60.0, 40.0, 60.0]:
                sensor.get_neural_state(cpu_percent=cpu, memory_percent=50.0)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0)
        # Range = 20 → cpu_stability = 1 - 20/40 = 0.5
        # delta = 0.5 * 0.7 + 1.0 * 0.3 = 0.65
        assert state.delta == pytest.approx(0.65, abs=0.05)

    def test_temp_variation_reduces_delta(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            for _ in range(5):
                sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0, cpu_temp=45.0)
            state = sensor.get_neural_state(cpu_percent=50.0, memory_percent=50.0, cpu_temp=55.0)
        assert state.delta < 1.0


class TestOutputRanges:
    """Test that all outputs are in valid [0, 1] range."""

    def test_extreme_inputs_stay_in_range(self, sensor):
        with patch("anima_mcp.computational_neural.psutil") as mock_ps:
            _mock_psutil(mock_ps)
            for cpu in [0, 50, 100, 200]:
                for mem in [0, 50, 100]:
                    state = sensor.get_neural_state(cpu_percent=cpu, memory_percent=mem)
                    for band in [state.delta, state.theta, state.alpha, state.beta, state.gamma]:
                        assert 0.0 <= band <= 1.0, f"Band out of range: {band} (cpu={cpu}, mem={mem})"
