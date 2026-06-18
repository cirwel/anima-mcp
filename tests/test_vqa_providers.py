"""
Tests for VQA (Visual Question Answering) providers in self_schema_renderer.

Validates provider selection, API response parsing, and fallback behavior.
"""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

from anima_mcp.self_schema_renderer import (
    evaluate_vqa,
    _call_vision_provider,
    _parse_vqa_response,
    _VQA_PROVIDERS,
    compute_visual_integrity_stub,
)
from anima_mcp.self_schema import SelfSchema, SchemaNode, SchemaEdge


def _make_provider_config(name: str, api_key: str = "test_key") -> dict:
    """Build a provider config dict for testing."""
    for cfg in _VQA_PROVIDERS:
        if cfg["name"] == name:
            return {**cfg, "api_key": api_key}
    raise ValueError(f"Unknown provider: {name}")


# === Test Fixtures ===

@pytest.fixture
def sample_ground_truth():
    """Sample VQA ground truth questions."""
    return [
        {"question": "How many nodes are in this graph?", "answer": "8", "type": "counting"},
        {"question": "Is there a gold node in the center?", "answer": "yes", "type": "existence"},
        {"question": "Are there blue nodes?", "answer": "yes", "type": "existence"},
        {"question": "How many green nodes are there?", "answer": "3", "type": "counting"},
        {"question": "What color is the center node?", "answer": "gold", "type": "attribute"},
    ]


@pytest.fixture
def sample_schema():
    """Create a sample self-schema for testing (matches extract_self_schema ID convention)."""
    from datetime import datetime

    nodes = [
        SchemaNode(node_id="identity", node_type="identity", label="Lumen", value=1.0),
        SchemaNode(node_id="anima_warmth", node_type="anima", label="Warmth", value=0.7),
        SchemaNode(node_id="anima_clarity", node_type="anima", label="Clarity", value=0.8),
        SchemaNode(node_id="anima_stability", node_type="anima", label="Stability", value=0.6),
        SchemaNode(node_id="anima_presence", node_type="anima", label="Presence", value=0.9),
        SchemaNode(node_id="sensor_light", node_type="sensor", label="Light", value=0.5),
        SchemaNode(node_id="sensor_temp", node_type="sensor", label="Temp", value=0.6),
        SchemaNode(node_id="sensor_humidity", node_type="sensor", label="Humid", value=0.4),
    ]
    edges = [
        SchemaEdge(source_id="identity", target_id="anima_warmth", weight=1.0),
        SchemaEdge(source_id="identity", target_id="anima_clarity", weight=1.0),
        SchemaEdge(source_id="sensor_light", target_id="anima_clarity", weight=0.5),
    ]
    return SelfSchema(timestamp=datetime.now(), nodes=nodes, edges=edges)


@pytest.fixture
def temp_png_file(tmp_path):
    """Create a temporary PNG file for testing."""
    # Create a minimal valid PNG (1x1 black pixel)
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0x0F, 0x00, 0x00,
        0x01, 0x01, 0x00, 0x05, 0x18, 0xD8, 0x4D, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,  # IEND chunk
        0x42, 0x60, 0x82,
    ])
    png_path = tmp_path / "test_schema.png"
    png_path.write_bytes(png_data)
    return png_path


# === Test Provider Selection ===

class TestProviderSelection:
    """Tests for VQA provider selection logic."""

    @pytest.mark.asyncio
    async def test_no_providers_returns_error(self, temp_png_file, sample_ground_truth):
        """When no API keys are set, should return error with stub_fallback flag."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all provider keys
            for key in ["GROQ_API_KEY", "HF_TOKEN", "TOGETHER_API_KEY", "ANTHROPIC_API_KEY"]:
                os.environ.pop(key, None)

            result = await evaluate_vqa(temp_png_file, sample_ground_truth)

            assert result["v_f"] is None
            assert "error" in result
            assert result["stub_fallback"] is True
            assert "groq.com" in result["error"]

    @pytest.mark.asyncio
    async def test_groq_is_first_priority(self, temp_png_file, sample_ground_truth):
        """Groq should be tried first when GROQ_API_KEY is set."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            with patch("anima_mcp.self_schema_renderer._call_vision_provider") as mock_call:
                mock_call.return_value = "1. 8\n2. yes\n3. yes\n4. 3\n5. gold"

                await evaluate_vqa(temp_png_file, sample_ground_truth)

                mock_call.assert_called()
                first_config = mock_call.call_args_list[0][0][0]
                assert first_config["name"] == "groq"

    @pytest.mark.asyncio
    async def test_together_used_when_groq_unavailable(self, temp_png_file, sample_ground_truth):
        """Together should be used when Groq/HF are not configured."""
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "test_key"}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            os.environ.pop("HF_TOKEN", None)

            with patch("anima_mcp.self_schema_renderer._call_vision_provider") as mock_call:
                mock_call.return_value = "1. 8\n2. yes\n3. yes\n4. 3\n5. gold"

                await evaluate_vqa(temp_png_file, sample_ground_truth)

                first_config = mock_call.call_args_list[0][0][0]
                assert first_config["name"] == "together"


# === Test Response Parsing ===

class TestResponseParsing:
    """Tests for VQA response parsing."""

    def test_parse_numbered_responses(self, sample_ground_truth):
        """Should correctly parse numbered responses."""
        response = "1. 8\n2. yes\n3. yes\n4. 3\n5. gold"

        result = _parse_vqa_response(response, sample_ground_truth, "test_model")

        assert result["v_f"] == 1.0  # All correct
        assert result["correct_count"] == 5
        assert result["total_count"] == 5
        assert result["stub"] is False

    def test_parse_partial_correct(self, sample_ground_truth):
        """Should handle partially correct responses."""
        response = "1. 8\n2. no\n3. yes\n4. 5\n5. blue"  # 2/5 correct

        result = _parse_vqa_response(response, sample_ground_truth, "test_model")

        assert result["v_f"] == 0.4  # 2/5 = 0.4
        assert result["correct_count"] == 2

    def test_parse_with_periods_in_answer(self, sample_ground_truth):
        """Should handle responses with periods after numbers."""
        response = "1. 8.\n2. yes.\n3. yes.\n4. 3.\n5. gold."

        result = _parse_vqa_response(response, sample_ground_truth, "test_model")

        # Should still match (answers contain expected)
        assert result["correct_count"] >= 3

    def test_parse_empty_response(self, sample_ground_truth):
        """Should handle empty or malformed responses gracefully."""
        response = ""

        result = _parse_vqa_response(response, sample_ground_truth, "test_model")

        assert result["v_f"] == 0.0
        assert result["correct_count"] == 0


# === Test Stub Computation ===

class TestStubComputation:
    """Tests for visual integrity stub computation."""

    def test_stub_returns_expected_fields(self, sample_schema):
        """Stub should return all expected fields."""
        from anima_mcp.self_schema_renderer import render_schema_to_pixels

        pixels = render_schema_to_pixels(sample_schema)
        result = compute_visual_integrity_stub(pixels, sample_schema)

        assert "v_f" in result
        assert "v_c" in result
        assert "V" in result
        assert result["stub"] is True
        assert 0 <= result["V"] <= 1

    def test_stub_with_empty_schema(self):
        """Stub should handle empty schema gracefully."""
        from datetime import datetime
        empty_schema = SelfSchema(timestamp=datetime.now(), nodes=[], edges=[])

        result = compute_visual_integrity_stub({}, empty_schema)

        assert result["V"] == 0.0
        assert result["stub"] is True

    def test_stub_reasonable_score(self, sample_schema):
        """Stub should return reasonable score for valid render."""
        from anima_mcp.self_schema_renderer import render_schema_to_pixels

        pixels = render_schema_to_pixels(sample_schema)
        result = compute_visual_integrity_stub(pixels, sample_schema)

        # Should be reasonably high for a proper render
        assert result["V"] > 0.5


# === Test API Call Mocking ===

class TestAPICallMocking:
    """Tests for mocked API calls to providers."""

    @pytest.mark.asyncio
    async def test_together_success_response(self):
        """Test successful Together AI API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "1. 8\n2. yes"}}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            config = _make_provider_config("together")
            result = await _call_vision_provider(config, "base64data", "test prompt")

            assert result == "1. 8\n2. yes"

    @pytest.mark.asyncio
    async def test_together_error_raises(self):
        """Test Together AI API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request: Invalid model"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception) as exc_info:
                config = _make_provider_config("together")
                await _call_vision_provider(config, "base64data", "test prompt")

            assert "together API error 400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_provider_error_raises_for_fallback(self):
        """Test that provider errors raise exceptions to trigger fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception) as exc_info:
                config = _make_provider_config("huggingface")
                await _call_vision_provider(config, "base64data", "test prompt")

            assert "503" in str(exc_info.value)
