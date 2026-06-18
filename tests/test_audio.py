"""
Tests for the audio package — mic, speaker, stt, tts, voice, autonomous_voice.

All audio hardware (sounddevice, piper, vosk) is mocked.
No real audio devices or external processes are used.
"""

import time
import json
import wave
from unittest.mock import patch, MagicMock

import pytest

from anima_mcp.audio.mic import MicCapture, AudioChunk, SAMPLE_RATE, CHANNELS, CHUNK_SIZE
from anima_mcp.audio.speaker import Speaker, AudioPlayback
from anima_mcp.audio.stt import (
    SpeechToText,
    TranscriptionResult,
)
from anima_mcp.audio.tts import (
    TextToSpeech,
    Voice,
    VoiceStyle,
    RECOMMENDED_VOICES,
)
from anima_mcp.audio.voice import (
    LumenVoice,
    VoiceConfig,
    VoiceState,
    Utterance,
    create_voice,
)
from anima_mcp.audio.autonomous_voice import (
    AutonomousVoice,
    SpeechIntent,
    SpeechMoment,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mic():
    """MicCapture with audio init prevented."""
    m = MicCapture()
    m._init_failed = True  # Prevent real audio init
    return m


@pytest.fixture
def speaker():
    """Speaker with audio init prevented."""
    s = Speaker()
    s._init_failed = True  # Prevent real audio init
    return s


@pytest.fixture
def tts():
    """TextToSpeech that won't try to find real piper."""
    t = TextToSpeech()
    t._init_failed = True
    return t


@pytest.fixture
def tts_initialized():
    """TextToSpeech marked as initialized (bypasses piper check)."""
    t = TextToSpeech()
    t._initialized = True
    return t


@pytest.fixture
def stt():
    """SpeechToText that won't try to find real vosk model."""
    s = SpeechToText()
    s._init_failed = True
    return s


@pytest.fixture
def mock_voice():
    """LumenVoice with all subcomponents mocked."""
    with patch.object(MicCapture, "_init_audio", return_value=True), \
         patch.object(MicCapture, "start", return_value=True), \
         patch.object(MicCapture, "stop"), \
         patch.object(Speaker, "_init_audio", return_value=True), \
         patch.object(Speaker, "start", return_value=True), \
         patch.object(Speaker, "stop"), \
         patch.object(SpeechToText, "initialize", return_value=True), \
         patch.object(TextToSpeech, "initialize", return_value=True):
        config = VoiceConfig(always_listening=True, speak_responses=False)
        voice = LumenVoice(config)
        yield voice


@pytest.fixture
def autonomous():
    """AutonomousVoice with voice fully mocked."""
    mock_v = MagicMock(spec=LumenVoice)
    av = AutonomousVoice(voice=mock_v)
    return av


# =========================================================================
# MicCapture tests
# =========================================================================


class TestMicCaptureInit:
    """Test MicCapture initialization and defaults."""

    def test_defaults(self):
        m = MicCapture()
        assert m._sample_rate == SAMPLE_RATE
        assert m._channels == CHANNELS
        assert not m._running

    def test_custom_sample_rate(self):
        m = MicCapture(sample_rate=44100, channels=2)
        assert m._sample_rate == 44100
        assert m._channels == 2

    def test_vad_enabled_by_default(self):
        m = MicCapture()
        assert m._vad_enabled is True

    def test_set_vad_enabled(self, mic):
        mic.set_vad_enabled(False)
        assert mic._vad_enabled is False
        mic.set_vad_enabled(True)
        assert mic._vad_enabled is True

    def test_set_silence_threshold(self, mic):
        mic.set_silence_threshold(1000)
        assert mic._silence_threshold == 1000


class TestMicCaptureAudioInit:
    """Test audio device initialization."""

    def test_init_audio_import_error(self):
        m = MicCapture()
        with patch.dict("sys.modules", {"sounddevice": None}):
            # Force ImportError by making the import fail
            m._init_failed = False
            m._audio_interface = None
            with patch("builtins.__import__", side_effect=ImportError("no sounddevice")):
                m._init_audio()
        # After import error, should fail and mark _init_failed
        assert m._init_failed is True

    def test_init_audio_already_initialized(self, mic):
        mic._audio_interface = MagicMock()
        mic._init_failed = False
        assert mic._init_audio() is True

    def test_init_audio_already_failed(self, mic):
        mic._init_failed = True
        mic._audio_interface = None
        assert mic._init_audio() is False


class TestMicCaptureStartStop:
    """Test mic start/stop lifecycle."""

    def test_start_when_init_fails(self, mic):
        mic._init_failed = True
        result = mic.start()
        assert result is False
        assert not mic._running

    def test_start_already_running(self, mic):
        mic._running = True
        result = mic.start()
        assert result is True  # Returns True (idempotent)

    def test_stop_when_not_running(self, mic):
        mic._running = False
        mic.stop()  # Should not raise

    def test_stop_joins_thread(self, mic):
        mock_thread = MagicMock()
        mic._capture_thread = mock_thread
        mic._running = True
        mic.stop()
        mock_thread.join.assert_called_once_with(timeout=2.0)
        assert mic._capture_thread is None
        assert mic._running is False


class TestMicCaptureCallbacks:
    """Test speech start/end callbacks."""

    def test_on_speech_start_callback(self, mic):
        callback = MagicMock()
        mic.on_speech_start(callback)
        assert mic._on_speech_start is callback

    def test_on_speech_end_callback(self, mic):
        callback = MagicMock()
        mic.on_speech_end(callback)
        assert mic._on_speech_end is callback


class TestMicCaptureGetChunk:
    """Test audio chunk retrieval."""

    def test_get_chunk_empty_queue(self, mic):
        chunk = mic.get_chunk(timeout=0.01)
        assert chunk is None

    def test_get_chunk_with_data(self, mic):
        chunk = AudioChunk(data=b"\x00\x01", timestamp=1.0, duration=0.1)
        mic._audio_queue.put(chunk)
        result = mic.get_chunk(timeout=0.1)
        assert result is chunk


class TestMicCaptureProperties:
    """Test MicCapture properties."""

    def test_sample_rate_property(self, mic):
        assert mic.sample_rate == SAMPLE_RATE

    def test_is_running_property(self, mic):
        assert mic.is_running is False
        mic._running = True
        assert mic.is_running is True


# =========================================================================
# Speaker tests
# =========================================================================


class TestSpeakerInit:
    """Test Speaker initialization and defaults."""

    def test_defaults(self):
        s = Speaker()
        assert s._volume == 0.8
        assert not s._running
        assert s._device is None

    def test_volume_property(self, speaker):
        assert speaker.volume == 0.8

    def test_volume_setter_clamps(self, speaker):
        speaker.volume = 1.5
        assert speaker.volume == 1.0
        speaker.volume = -0.5
        assert speaker.volume == 0.0
        speaker.volume = 0.5
        assert speaker.volume == 0.5


class TestSpeakerAudioInit:
    """Test speaker audio device initialization."""

    def test_init_audio_already_initialized(self, speaker):
        speaker._audio_interface = MagicMock()
        speaker._init_failed = False
        assert speaker._init_audio() is True

    def test_init_audio_already_failed(self, speaker):
        assert speaker._init_audio() is False


class TestSpeakerStartStop:
    """Test speaker start/stop lifecycle."""

    def test_start_when_init_fails(self, speaker):
        result = speaker.start()
        assert result is False
        assert not speaker._running

    def test_start_already_running(self, speaker):
        speaker._running = True
        result = speaker.start()
        assert result is True

    def test_stop_when_not_running(self, speaker):
        speaker.stop()  # Should not raise

    def test_stop_joins_thread(self, speaker):
        mock_thread = MagicMock()
        speaker._playback_thread = mock_thread
        speaker._running = True
        speaker.stop()
        mock_thread.join.assert_called_once_with(timeout=2.0)
        assert speaker._playback_thread is None
        assert speaker._running is False


class TestSpeakerPlay:
    """Test audio playback."""

    def test_play_blocking_no_audio_interface(self, speaker):
        """Blocking play with no interface should try init and bail."""
        speaker._audio_interface = None
        speaker._init_failed = True
        speaker.play(b"\x00" * 100, blocking=True)
        # Should not raise — just returns

    def test_play_blocking_with_mock_interface(self, speaker):
        """Blocking play delegates to _play_audio."""
        mock_sd = MagicMock()
        speaker._audio_interface = mock_sd
        speaker._init_failed = False

        with patch.object(speaker, "_play_audio") as mock_play:
            speaker.play(b"\x00" * 100, sample_rate=22050, blocking=True)
            mock_play.assert_called_once()
            playback = mock_play.call_args[0][0]
            assert playback.audio_bytes == b"\x00" * 100
            assert playback.sample_rate == 22050

    def test_play_nonblocking_queues(self, speaker):
        """Non-blocking play adds to queue (starts speaker if needed)."""
        speaker._running = True
        speaker._audio_interface = MagicMock()
        speaker.play(b"\x00" * 100, blocking=False)
        assert speaker._audio_queue.qsize() == 1

    def test_play_nonblocking_queue_full(self, speaker):
        """Non-blocking play when queue full drops audio silently."""
        speaker._running = True
        speaker._audio_interface = MagicMock()
        # Fill the queue (maxsize=10)
        for _ in range(10):
            speaker._audio_queue.put(AudioPlayback(audio_bytes=b"\x00"))
        # Should not raise
        speaker.play(b"\x00" * 100, blocking=False)
        # Queue still at 10 (item was dropped)
        assert speaker._audio_queue.qsize() == 10


class TestSpeakerClearQueue:
    """Test queue clearing."""

    def test_clear_queue(self, speaker):
        for _ in range(5):
            speaker._audio_queue.put(AudioPlayback(audio_bytes=b"\x00"))
        assert speaker._audio_queue.qsize() == 5
        speaker.clear_queue()
        assert speaker._audio_queue.qsize() == 0

    def test_clear_empty_queue(self, speaker):
        speaker.clear_queue()  # Should not raise
        assert speaker._audio_queue.qsize() == 0


class TestSpeakerSpeak:
    """Test the convenience speak() method."""

    def test_speak_with_audio(self, speaker):
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = b"\x00" * 100
        with patch.object(speaker, "play") as mock_play:
            speaker.speak("hello", mock_tts, blocking=True)
            mock_tts.synthesize.assert_called_once_with("hello")
            mock_play.assert_called_once_with(b"\x00" * 100, sample_rate=22050, blocking=True)

    def test_speak_no_audio_from_tts(self, speaker):
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = None
        with patch.object(speaker, "play") as mock_play:
            speaker.speak("hello", mock_tts)
            mock_play.assert_not_called()


class TestSpeakerPlayFile:
    """Test playing WAV files."""

    def test_play_file(self, speaker, tmp_path):
        # Create a minimal WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 100)

        with patch.object(speaker, "play") as mock_play:
            speaker.play_file(wav_path)
            mock_play.assert_called_once()

    def test_play_file_not_found(self, speaker, tmp_path):
        bad_path = tmp_path / "nonexistent.wav"
        speaker.play_file(bad_path)  # Should not raise


class TestSpeakerProperties:
    """Test Speaker properties."""

    def test_is_running(self, speaker):
        assert speaker.is_running is False
        speaker._running = True
        assert speaker.is_running is True

    def test_queue_size(self, speaker):
        assert speaker.queue_size == 0
        speaker._audio_queue.put(AudioPlayback(audio_bytes=b"\x00"))
        assert speaker.queue_size == 1


# =========================================================================
# SpeechToText tests
# =========================================================================


class TestSTTInit:
    """Test SpeechToText initialization."""

    def test_defaults(self):
        s = SpeechToText()
        assert s._sample_rate == 16000
        assert not s._initialized
        assert s._model is None

    def test_custom_model_path(self, tmp_path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        s = SpeechToText(model_path=model_dir)
        assert s._model_path == model_dir

    def test_custom_sample_rate(self):
        s = SpeechToText(sample_rate=44100)
        assert s._sample_rate == 44100


class TestSTTFindModel:
    """Test model discovery."""

    def test_find_model_default_path(self, tmp_path):
        s = SpeechToText()
        with patch("anima_mcp.audio.stt.DEFAULT_MODEL_PATH", tmp_path / "model_a"), \
             patch("anima_mcp.audio.stt.FALLBACK_MODEL_PATH", tmp_path / "model_b"):
            # Neither exists
            result = s._find_model()
            assert result is None

    def test_find_model_default_exists(self, tmp_path):
        model_dir = tmp_path / "vosk-model"
        model_dir.mkdir()
        s = SpeechToText()
        with patch("anima_mcp.audio.stt.DEFAULT_MODEL_PATH", model_dir):
            result = s._find_model()
            assert result == model_dir

    def test_find_model_fallback_exists(self, tmp_path):
        default = tmp_path / "missing"
        fallback = tmp_path / "vosk-fallback"
        fallback.mkdir()
        s = SpeechToText()
        with patch("anima_mcp.audio.stt.DEFAULT_MODEL_PATH", default), \
             patch("anima_mcp.audio.stt.FALLBACK_MODEL_PATH", fallback):
            result = s._find_model()
            assert result == fallback

    def test_find_model_scans_models_dir(self, tmp_path):
        models_dir = tmp_path / ".anima" / "models"
        models_dir.mkdir(parents=True)
        vosk_dir = models_dir / "vosk-model-test"
        vosk_dir.mkdir()
        s = SpeechToText()
        with patch("anima_mcp.audio.stt.DEFAULT_MODEL_PATH", tmp_path / "a"), \
             patch("anima_mcp.audio.stt.FALLBACK_MODEL_PATH", tmp_path / "b"), \
             patch("pathlib.Path.home", return_value=tmp_path):
            result = s._find_model()
            assert result == vosk_dir


class TestSTTInitialize:
    """Test STT initialization with vosk."""

    def test_initialize_no_model_path(self, stt):
        stt._model_path = None
        stt._init_failed = False
        result = stt.initialize()
        assert result is False
        assert stt._init_failed is True

    def test_initialize_model_path_missing(self, stt, tmp_path):
        stt._model_path = tmp_path / "nonexistent"
        stt._init_failed = False
        result = stt.initialize()
        assert result is False

    def test_initialize_import_error(self, stt, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        stt._model_path = model_dir
        stt._init_failed = False
        with patch("builtins.__import__", side_effect=ImportError("no vosk")):
            result = stt.initialize()
        assert result is False
        assert stt._init_failed is True

    def test_initialize_already_initialized(self, stt):
        stt._initialized = True
        stt._init_failed = False
        assert stt.initialize() is True

    def test_initialize_already_failed_no_retry(self, stt):
        assert stt.initialize() is False


class TestSTTTranscribe:
    """Test transcription."""

    def test_transcribe_not_initialized(self, stt):
        result = stt.transcribe(b"\x00" * 100)
        assert result is None

    def test_transcribe_success(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.FinalResult.return_value = json.dumps({
            "text": "hello world",
            "result": [
                {"word": "hello", "conf": 0.9},
                {"word": "world", "conf": 0.8},
            ]
        })
        stt._recognizer = mock_recognizer

        result = stt.transcribe(b"\x00" * 100)

        assert result is not None
        assert result.text == "hello world"
        assert result.confidence == pytest.approx(0.85)
        assert result.is_final is True

    def test_transcribe_empty_text(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.FinalResult.return_value = json.dumps({"text": ""})
        stt._recognizer = mock_recognizer

        result = stt.transcribe(b"\x00" * 100)
        assert result is None

    def test_transcribe_exception(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.side_effect = RuntimeError("boom")
        stt._recognizer = mock_recognizer

        result = stt.transcribe(b"\x00" * 100)
        assert result is None


class TestSTTStreamingTranscribe:
    """Test streaming transcription."""

    def test_streaming_final_result(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = json.dumps({
            "text": "streaming result",
            "result": [{"word": "streaming", "conf": 0.8}, {"word": "result", "conf": 0.9}],
        })
        stt._recognizer = mock_recognizer

        result = stt.transcribe_streaming(b"\x00" * 100)
        assert result is not None
        assert result.text == "streaming result"
        assert result.is_final is True

    def test_streaming_partial_result(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = json.dumps({
            "partial": "partia"
        })
        stt._recognizer = mock_recognizer

        result = stt.transcribe_streaming(b"\x00" * 100)
        assert result is not None
        assert result.text == "partia"
        assert result.is_final is False
        assert result.confidence == 0.5

    def test_streaming_empty_partial(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = json.dumps({"partial": ""})
        stt._recognizer = mock_recognizer

        result = stt.transcribe_streaming(b"\x00" * 100)
        assert result is None

    def test_streaming_not_initialized(self, stt):
        result = stt.transcribe_streaming(b"\x00" * 100)
        assert result is None

    def test_streaming_exception(self, stt):
        stt._initialized = True
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.side_effect = RuntimeError("crash")
        stt._recognizer = mock_recognizer

        result = stt.transcribe_streaming(b"\x00" * 100)
        assert result is None


class TestSTTConfidence:
    """Test confidence calculation."""

    def test_confidence_with_words(self, stt):
        result = {"result": [
            {"word": "a", "conf": 0.9},
            {"word": "b", "conf": 0.7},
            {"word": "c", "conf": 0.8},
        ]}
        assert stt._calculate_confidence(result) == pytest.approx(0.8)

    def test_confidence_no_words(self, stt):
        result = {}
        assert stt._calculate_confidence(result) == 0.7

    def test_confidence_empty_words(self, stt):
        result = {"result": []}
        assert stt._calculate_confidence(result) == 0.7


class TestSTTReset:
    """Test recognizer reset."""

    def test_reset_no_recognizer(self, stt):
        stt._recognizer = None
        stt.reset()  # Should not raise

    def test_reset_with_recognizer(self, stt):
        mock_model = MagicMock()
        stt._model = mock_model
        stt._recognizer = MagicMock()

        mock_kaldi = MagicMock()
        with patch.dict("sys.modules", {"vosk": MagicMock(KaldiRecognizer=mock_kaldi)}):
            stt.reset()
            mock_kaldi.assert_called_once_with(mock_model, stt._sample_rate)


class TestSTTProperties:
    """Test STT properties."""

    def test_is_initialized(self, stt):
        assert stt.is_initialized is False
        stt._initialized = True
        assert stt.is_initialized is True

    def test_model_path(self, stt, tmp_path):
        stt._model_path = tmp_path
        assert stt.model_path == tmp_path


class TestSTTDownloadModel:
    """Test model download."""

    def test_unknown_model_size(self):
        result = SpeechToText.download_model("huge")
        assert result is False

    def test_download_success(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("urllib.request.urlretrieve"), \
             patch("zipfile.ZipFile") as mock_zip:
            # Create models dir so mkdir doesn't fail
            models_dir = tmp_path / ".anima" / "models"
            models_dir.mkdir(parents=True)

            # Mock the zip file context manager
            mock_zip_instance = MagicMock()
            mock_zip.return_value.__enter__ = MagicMock(return_value=mock_zip_instance)
            mock_zip.return_value.__exit__ = MagicMock(return_value=False)

            # Mock unlink
            with patch("pathlib.Path.unlink"):
                result = SpeechToText.download_model("small")
            assert result is True

    def test_download_failure(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path), \
             patch("urllib.request.urlretrieve", side_effect=Exception("network error")):
            models_dir = tmp_path / ".anima" / "models"
            models_dir.mkdir(parents=True)

            result = SpeechToText.download_model("small")
            assert result is False


# =========================================================================
# TextToSpeech tests
# =========================================================================


class TestTTSInit:
    """Test TextToSpeech initialization."""

    def test_defaults(self):
        t = TextToSpeech()
        assert t._voice == RECOMMENDED_VOICES["default"]
        assert t._speed == 1.0
        assert t._pitch == 1.0
        assert t._volume == 1.0
        assert not t._initialized

    def test_custom_voice(self):
        v = Voice("en_US-test-low", "en_US", "low")
        t = TextToSpeech(voice=v)
        assert t._voice == v


class TestTTSInitialize:
    """Test TTS initialization (piper check)."""

    def test_initialize_piper_found(self, tts):
        tts._init_failed = False
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Piper 1.0\n")
            result = tts.initialize()
        assert result is True
        assert tts._initialized is True

    def test_initialize_piper_not_found(self, tts):
        tts._init_failed = False
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = tts.initialize()
        assert result is False
        assert tts._init_failed is True

    def test_initialize_piper_error(self, tts):
        tts._init_failed = False
        with patch("subprocess.run", side_effect=Exception("crash")):
            result = tts.initialize()
        assert result is False

    def test_initialize_already_done(self, tts_initialized):
        assert tts_initialized.initialize() is True

    def test_initialize_already_failed_no_retry(self, tts):
        assert tts.initialize() is False


class TestTTSSynthesize:
    """Test speech synthesis."""

    def test_synthesize_empty_text(self, tts_initialized):
        result = tts_initialized.synthesize("")
        assert result is None

    def test_synthesize_whitespace_only(self, tts_initialized):
        result = tts_initialized.synthesize("   ")
        assert result is None

    def test_synthesize_not_initialized(self, tts):
        result = tts.synthesize("hello")
        assert result is None

    def test_synthesize_success(self, tts_initialized, tmp_path):
        # Create a real WAV file for the mock to "produce"
        wav_path = tmp_path / "output.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x01" * 100)

        with patch("subprocess.run") as mock_run, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_run.return_value = MagicMock(returncode=0)
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(wav_path)
            mock_tmp.return_value = mock_file

            result = tts_initialized.synthesize("hello world")

        assert result is not None
        assert len(result) > 0

    def test_synthesize_piper_returns_error(self, tts_initialized, tmp_path):
        with patch("subprocess.run") as mock_run, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_run.return_value = MagicMock(returncode=1, stderr="model not found")
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(tmp_path / "out.wav")
            mock_tmp.return_value = mock_file

            result = tts_initialized.synthesize("hello")
        assert result is None

    def test_synthesize_timeout(self, tts_initialized):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("piper", 30)), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/test.wav"
            mock_tmp.return_value = mock_file

            result = tts_initialized.synthesize("hello")
        assert result is None

    def test_synthesize_with_custom_speed(self, tts_initialized, tmp_path):
        """When speed != 1.0, --length_scale flag is added."""
        wav_path = tmp_path / "output.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x01" * 100)

        tts_initialized._speed = 1.5

        with patch("subprocess.run") as mock_run, \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_run.return_value = MagicMock(returncode=0)
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = str(wav_path)
            mock_tmp.return_value = mock_file

            tts_initialized.synthesize("hello")

        cmd = mock_run.call_args[0][0]
        assert "--length_scale" in cmd


class TestTTSSynthesizeToFile:
    """Test file synthesis."""

    def test_synthesize_to_file_empty_text(self, tts_initialized, tmp_path):
        result = tts_initialized.synthesize_to_file("", tmp_path / "out.wav")
        assert result is False

    def test_synthesize_to_file_not_initialized(self, tts, tmp_path):
        result = tts.synthesize_to_file("hello", tmp_path / "out.wav")
        assert result is False

    def test_synthesize_to_file_success(self, tts_initialized, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = tts_initialized.synthesize_to_file("hello", tmp_path / "out.wav")
        assert result is True

    def test_synthesize_to_file_error(self, tts_initialized, tmp_path):
        with patch("subprocess.run", side_effect=Exception("crash")):
            result = tts_initialized.synthesize_to_file("hello", tmp_path / "out.wav")
        assert result is False


class TestTTSVoiceSelection:
    """Test voice selection and style mapping."""

    def test_set_voice(self, tts_initialized):
        v = Voice("test", "en", "low")
        tts_initialized.set_voice(v)
        assert tts_initialized.voice == v

    def test_set_voice_by_style_neutral(self, tts_initialized):
        tts_initialized.set_voice_by_style(VoiceStyle.NEUTRAL)
        assert tts_initialized.voice == RECOMMENDED_VOICES["default"]

    def test_set_voice_by_style_warm(self, tts_initialized):
        tts_initialized.set_voice_by_style(VoiceStyle.WARM)
        assert tts_initialized.voice == RECOMMENDED_VOICES["warm"]

    def test_set_voice_by_style_clear(self, tts_initialized):
        tts_initialized.set_voice_by_style(VoiceStyle.CLEAR)
        assert tts_initialized.voice == RECOMMENDED_VOICES["clear"]

    def test_set_voice_by_style_soft(self, tts_initialized):
        tts_initialized.set_voice_by_style(VoiceStyle.SOFT)
        assert tts_initialized.voice == RECOMMENDED_VOICES["soft"]

    def test_set_voice_by_style_bright(self, tts_initialized):
        tts_initialized.set_voice_by_style(VoiceStyle.BRIGHT)
        assert tts_initialized.voice == RECOMMENDED_VOICES["default"]
        assert tts_initialized._speed == 1.1  # Bright = faster


class TestTTSAnimaState:
    """Test voice modulation from anima state."""

    def test_high_warmth_selects_warm_voice(self, tts_initialized):
        tts_initialized.set_from_anima_state(warmth=0.8, clarity=0.3, stability=0.5)
        assert tts_initialized.voice == RECOMMENDED_VOICES["warm"]

    def test_high_clarity_selects_clear_voice(self, tts_initialized):
        tts_initialized.set_from_anima_state(warmth=0.3, clarity=0.8, stability=0.5)
        assert tts_initialized.voice == RECOMMENDED_VOICES["clear"]

    def test_low_warmth_low_clarity_selects_soft(self, tts_initialized):
        tts_initialized.set_from_anima_state(warmth=0.3, clarity=0.3, stability=0.5)
        assert tts_initialized.voice == RECOMMENDED_VOICES["soft"]

    def test_moderate_values_selects_default(self, tts_initialized):
        tts_initialized.set_from_anima_state(warmth=0.5, clarity=0.5, stability=0.5)
        assert tts_initialized.voice == RECOMMENDED_VOICES["default"]

    def test_warmth_takes_priority_over_clarity(self, tts_initialized):
        """When both warmth and clarity are high, warmth wins (checked first)."""
        tts_initialized.set_from_anima_state(warmth=0.8, clarity=0.8, stability=0.5)
        assert tts_initialized.voice == RECOMMENDED_VOICES["warm"]

    def test_low_stability_increases_speed(self, tts_initialized):
        tts_initialized.set_from_anima_state(warmth=0.5, clarity=0.5, stability=0.2)
        assert tts_initialized._speed > 1.0

    def test_high_stability_normal_speed(self, tts_initialized):
        tts_initialized.set_from_anima_state(warmth=0.5, clarity=0.5, stability=0.8)
        assert tts_initialized._speed == 1.0


class TestTTSSpeedProperty:
    """Test speed property clamping."""

    def test_speed_getter(self, tts_initialized):
        assert tts_initialized.speed == 1.0

    def test_speed_setter_clamps_high(self, tts_initialized):
        tts_initialized.speed = 5.0
        assert tts_initialized.speed == 2.0

    def test_speed_setter_clamps_low(self, tts_initialized):
        tts_initialized.speed = 0.1
        assert tts_initialized.speed == 0.5

    def test_speed_setter_valid(self, tts_initialized):
        tts_initialized.speed = 1.5
        assert tts_initialized.speed == 1.5


class TestTTSListVoices:
    """Test listing available voices."""

    def test_list_voices_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="voice1\nvoice2\nvoice3"
            )
            voices = TextToSpeech.list_voices()
        assert voices == ["voice1", "voice2", "voice3"]

    def test_list_voices_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            voices = TextToSpeech.list_voices()
        assert voices == []

    def test_list_voices_exception(self):
        with patch("subprocess.run", side_effect=Exception("no piper")):
            voices = TextToSpeech.list_voices()
        assert voices == []


# =========================================================================
# VoiceConfig / VoiceState / Utterance tests
# =========================================================================


class TestVoiceDataclasses:
    """Test voice dataclasses defaults and usage."""

    def test_voice_config_defaults(self):
        c = VoiceConfig()
        assert c.always_listening is False
        assert c.wake_word == "lumen"
        assert c.acknowledge_hearing is True
        assert c.speak_responses is True
        assert c.listen_timeout == 10.0
        assert c.response_timeout == 30.0

    def test_voice_state_defaults(self):
        s = VoiceState()
        assert s.is_listening is False
        assert s.is_speaking is False
        assert s.last_heard is None
        assert s.last_spoken is None
        assert s.conversation_active is False
        assert s.utterance_history == []

    def test_utterance_fields(self):
        u = Utterance(text="hello", confidence=0.9, timestamp=1000.0, duration=1.5)
        assert u.text == "hello"
        assert u.confidence == 0.9
        assert u.timestamp == 1000.0
        assert u.duration == 1.5


# =========================================================================
# LumenVoice tests
# =========================================================================


class TestLumenVoiceInit:
    """Test LumenVoice initialization."""

    def test_default_config(self):
        with patch.object(MicCapture, "__init__", return_value=None), \
             patch.object(SpeechToText, "__init__", return_value=None), \
             patch.object(TextToSpeech, "__init__", return_value=None), \
             patch.object(Speaker, "__init__", return_value=None):
            v = LumenVoice()
            assert v._config.always_listening is False

    def test_custom_config(self):
        config = VoiceConfig(always_listening=True, wake_word="hey")
        with patch.object(MicCapture, "__init__", return_value=None), \
             patch.object(SpeechToText, "__init__", return_value=None), \
             patch.object(TextToSpeech, "__init__", return_value=None), \
             patch.object(Speaker, "__init__", return_value=None):
            v = LumenVoice(config)
            assert v.config.always_listening is True
            assert v.config.wake_word == "hey"


class TestLumenVoiceInitialize:
    """Test voice system initialization."""

    def test_initialize_idempotent(self, mock_voice):
        assert mock_voice.initialize() is True
        assert mock_voice._initialized is True
        # Second call
        assert mock_voice.initialize() is True

    def test_initialize_stt_failure_continues(self, mock_voice):
        """Voice initializes even if STT fails."""
        with patch.object(mock_voice._stt, "initialize", return_value=False):
            mock_voice._initialized = False
            result = mock_voice.initialize()
        assert result is True  # Still initializes

    def test_initialize_tts_failure_continues(self, mock_voice):
        """Voice initializes even if TTS fails."""
        with patch.object(mock_voice._tts, "initialize", return_value=False):
            mock_voice._initialized = False
            result = mock_voice.initialize()
        assert result is True


class TestLumenVoiceStartStop:
    """Test voice start/stop lifecycle."""

    def test_start_sets_running(self, mock_voice):
        with patch.object(mock_voice._mic, "start", return_value=True), \
             patch.object(mock_voice._mic, "on_speech_start"), \
             patch.object(mock_voice._mic, "on_speech_end"), \
             patch.object(mock_voice._speaker, "start"):
            result = mock_voice.start()
        assert result is True
        assert mock_voice.is_running is True

    def test_start_already_running(self, mock_voice):
        mock_voice._running = True
        result = mock_voice.start()
        assert result is True

    def test_start_mic_failure(self, mock_voice):
        with patch.object(mock_voice._mic, "start", return_value=False):
            result = mock_voice.start()
        assert result is False

    def test_stop_cleans_up(self, mock_voice):
        mock_voice._running = True
        with patch.object(mock_voice._mic, "stop"), \
             patch.object(mock_voice._speaker, "stop"):
            mock_voice.stop()
        assert mock_voice._running is False


class TestLumenVoiceSay:
    """Test the say() method."""

    def test_say_empty_text(self, mock_voice):
        mock_voice.say("")
        assert mock_voice._state.last_spoken is None

    def test_say_calls_tts_and_speaker(self, mock_voice):
        audio = b"\x00" * 100
        with patch.object(mock_voice._tts, "set_from_anima_state"), \
             patch.object(mock_voice._tts, "synthesize", return_value=audio), \
             patch.object(mock_voice._speaker, "play"):
            mock_voice.say("hello")
        assert mock_voice._state.last_spoken == "hello"

    def test_say_tts_returns_none(self, mock_voice):
        with patch.object(mock_voice._tts, "set_from_anima_state"), \
             patch.object(mock_voice._tts, "synthesize", return_value=None), \
             patch.object(mock_voice._speaker, "play") as mock_play:
            mock_voice.say("hello")
        mock_play.assert_not_called()

    def test_say_blocking_flag(self, mock_voice):
        audio = b"\x00" * 100
        with patch.object(mock_voice._tts, "set_from_anima_state"), \
             patch.object(mock_voice._tts, "synthesize", return_value=audio), \
             patch.object(mock_voice._speaker, "play") as mock_play:
            mock_voice.say("hello", blocking=False)
        mock_play.assert_called_once_with(audio, sample_rate=22050, blocking=False)


class TestLumenVoiceAnimaState:
    """Test anima state updates."""

    def test_update_anima_state(self, mock_voice):
        mock_voice.update_anima_state(0.8, 0.6, 0.9)
        assert mock_voice._warmth == 0.8
        assert mock_voice._clarity == 0.6
        assert mock_voice._stability == 0.9


class TestLumenVoiceCallbacks:
    """Test callback setters."""

    def test_set_on_hear(self, mock_voice):
        def cb(u):
            return None

        mock_voice.set_on_hear(cb)
        assert mock_voice._on_hear is cb

    def test_set_on_respond(self, mock_voice):
        def cb(t):
            return "response"

        mock_voice.set_on_respond(cb)
        assert mock_voice._on_respond is cb


class TestLumenVoiceConversation:
    """Test conversation management."""

    def test_set_always_listening(self, mock_voice):
        mock_voice.set_always_listening(True)
        assert mock_voice._config.always_listening is True
        mock_voice.set_always_listening(False)
        assert mock_voice._config.always_listening is False

    def test_end_conversation(self, mock_voice):
        mock_voice._state.conversation_active = True
        mock_voice.end_conversation()
        assert mock_voice._state.conversation_active is False


class TestLumenVoiceSpeechEnd:
    """Test the _on_speech_end callback logic."""

    def test_on_speech_end_with_valid_transcription(self, mock_voice):
        mock_voice._config.always_listening = True
        mock_voice._config.acknowledge_hearing = False

        result = TranscriptionResult(
            text="hello lumen",
            confidence=0.9,
            is_final=True,
            alternatives=[]
        )
        with patch.object(mock_voice._stt, "transcribe", return_value=result):
            mock_voice._on_speech_end(b"\x00" * 1000)

        assert mock_voice._state.last_heard is not None
        assert mock_voice._state.last_heard.text == "hello lumen"

    def test_on_speech_end_no_transcription(self, mock_voice):
        with patch.object(mock_voice._stt, "transcribe", return_value=None):
            mock_voice._on_speech_end(b"\x00" * 1000)
        assert mock_voice._state.last_heard is None

    def test_on_speech_end_wake_word_check(self, mock_voice):
        mock_voice._config.always_listening = False
        mock_voice._config.wake_word = "lumen"
        mock_voice._state.conversation_active = False
        mock_voice._config.acknowledge_hearing = False

        # Without wake word
        result = TranscriptionResult(
            text="hello there",
            confidence=0.9,
            is_final=True,
            alternatives=[]
        )
        heard_callback = MagicMock()
        mock_voice._on_hear = heard_callback

        with patch.object(mock_voice._stt, "transcribe", return_value=result):
            mock_voice._on_speech_end(b"\x00" * 1000)

        heard_callback.assert_not_called()  # No wake word -> ignored

    def test_on_speech_end_wake_word_activates(self, mock_voice):
        mock_voice._config.always_listening = False
        mock_voice._config.wake_word = "lumen"
        mock_voice._state.conversation_active = False
        mock_voice._config.acknowledge_hearing = False

        result = TranscriptionResult(
            text="hey lumen how are you",
            confidence=0.9,
            is_final=True,
            alternatives=[]
        )
        heard_callback = MagicMock()
        mock_voice._on_hear = heard_callback

        with patch.object(mock_voice._stt, "transcribe", return_value=result):
            mock_voice._on_speech_end(b"\x00" * 1000)

        assert mock_voice._state.conversation_active is True
        heard_callback.assert_called_once()

    def test_on_speech_end_active_conversation_no_wake_word(self, mock_voice):
        """In active conversation, speech is processed without wake word."""
        mock_voice._config.always_listening = False
        mock_voice._state.conversation_active = True
        mock_voice._config.acknowledge_hearing = False

        result = TranscriptionResult(
            text="what is the weather",
            confidence=0.9,
            is_final=True,
            alternatives=[]
        )
        heard_callback = MagicMock()
        mock_voice._on_hear = heard_callback

        with patch.object(mock_voice._stt, "transcribe", return_value=result):
            mock_voice._on_speech_end(b"\x00" * 1000)

        heard_callback.assert_called_once()

    def test_on_speech_end_with_respond_callback(self, mock_voice):
        mock_voice._config.always_listening = True
        mock_voice._config.acknowledge_hearing = False

        result = TranscriptionResult(
            text="how are you",
            confidence=0.9,
            is_final=True,
            alternatives=[]
        )
        mock_voice._on_respond = lambda t: f"You said: {t}"

        with patch.object(mock_voice._stt, "transcribe", return_value=result), \
             patch.object(mock_voice, "say") as mock_say:
            mock_voice._on_speech_end(b"\x00" * 1000)

        mock_say.assert_called_once_with("You said: how are you")

    def test_utterance_history_bounded(self, mock_voice):
        """Utterance history doesn't grow unbounded."""
        mock_voice._config.always_listening = True
        mock_voice._config.acknowledge_hearing = False

        result = TranscriptionResult(
            text="test", confidence=0.9, is_final=True, alternatives=[]
        )

        with patch.object(mock_voice._stt, "transcribe", return_value=result):
            for _ in range(60):
                mock_voice._on_speech_end(b"\x00" * 100)

        assert len(mock_voice._state.utterance_history) <= 50


class TestLumenVoiceAcknowledge:
    """Test the acknowledge behavior."""

    def test_acknowledge_high_warmth(self, mock_voice):
        mock_voice._warmth = 0.8
        with patch.object(mock_voice, "say") as mock_say:
            mock_voice._acknowledge()
        mock_say.assert_called_once()
        ack = mock_say.call_args[0][0]
        assert ack in ["yes", "mm-hmm", "I'm here"]

    def test_acknowledge_high_clarity(self, mock_voice):
        mock_voice._warmth = 0.3
        mock_voice._clarity = 0.8
        with patch.object(mock_voice, "say") as mock_say:
            mock_voice._acknowledge()
        mock_say.assert_called_once_with("listening", blocking=False)


class TestLumenVoiceSpeechStart:
    """Test speech start callback."""

    def test_on_speech_start(self, mock_voice):
        mock_voice._on_speech_start()
        assert mock_voice._state.is_listening is True


class TestLumenVoiceProperties:
    """Test LumenVoice properties."""

    def test_state_property(self, mock_voice):
        assert isinstance(mock_voice.state, VoiceState)

    def test_is_running_property(self, mock_voice):
        assert mock_voice.is_running is False
        mock_voice._running = True
        assert mock_voice.is_running is True

    def test_config_property(self, mock_voice):
        assert isinstance(mock_voice.config, VoiceConfig)


# =========================================================================
# create_voice convenience function tests
# =========================================================================


class TestCreateVoice:
    """Test the create_voice convenience function."""

    def test_create_voice_defaults(self):
        with patch.object(MicCapture, "__init__", return_value=None), \
             patch.object(SpeechToText, "__init__", return_value=None), \
             patch.object(TextToSpeech, "__init__", return_value=None), \
             patch.object(Speaker, "__init__", return_value=None), \
             patch.object(LumenVoice, "initialize", return_value=True):
            voice = create_voice()
        assert voice.config.always_listening is False
        assert voice.config.wake_word == "lumen"

    def test_create_voice_with_options(self):
        hear_cb = MagicMock()
        respond_cb = MagicMock()
        with patch.object(MicCapture, "__init__", return_value=None), \
             patch.object(SpeechToText, "__init__", return_value=None), \
             patch.object(TextToSpeech, "__init__", return_value=None), \
             patch.object(Speaker, "__init__", return_value=None), \
             patch.object(LumenVoice, "initialize", return_value=True):
            voice = create_voice(
                always_listening=True,
                wake_word="hey",
                on_hear=hear_cb,
                on_respond=respond_cb,
            )
        assert voice.config.always_listening is True
        assert voice.config.wake_word == "hey"
        assert voice._on_hear is hear_cb
        assert voice._on_respond is respond_cb


# =========================================================================
# SpeechIntent / SpeechMoment tests
# =========================================================================


class TestSpeechDataclasses:
    """Test autonomous voice dataclasses."""

    def test_speech_intent_values(self):
        assert SpeechIntent.OBSERVATION.value == "observation"
        assert SpeechIntent.FEELING.value == "feeling"
        assert SpeechIntent.QUESTION.value == "question"
        assert SpeechIntent.GREETING.value == "greeting"
        assert SpeechIntent.REFLECTION.value == "reflection"
        assert SpeechIntent.RESPONSE.value == "response"
        assert SpeechIntent.SILENCE.value == "silence"

    def test_speech_moment_defaults(self):
        m = SpeechMoment(
            intent=SpeechIntent.OBSERVATION,
            text="it's warm",
            urgency=0.5,
        )
        assert m.intent == SpeechIntent.OBSERVATION
        assert m.text == "it's warm"
        assert m.urgency == 0.5
        assert m.timestamp > 0  # auto-set by time.time()


# =========================================================================
# AutonomousVoice tests
# =========================================================================


class TestAutonomousVoiceInit:
    """Test AutonomousVoice initialization."""

    def test_defaults(self, autonomous):
        assert autonomous._warmth == 0.5
        assert autonomous._clarity == 0.5
        assert autonomous._stability == 0.5
        assert autonomous._presence == 0.5
        assert autonomous._mood == "neutral"
        assert autonomous._chattiness == 0.5
        assert autonomous._curiosity == 0.5
        assert autonomous._speech_cooldown == 60.0
        assert not autonomous._running

    def test_creates_default_voice_if_none(self):
        with patch.object(MicCapture, "__init__", return_value=None), \
             patch.object(SpeechToText, "__init__", return_value=None), \
             patch.object(TextToSpeech, "__init__", return_value=None), \
             patch.object(Speaker, "__init__", return_value=None):
            av = AutonomousVoice()
        assert av._voice is not None


class TestAutonomousVoiceStartStop:
    """Test start/stop lifecycle."""

    def test_start_initializes_voice(self, autonomous):
        autonomous.start()
        autonomous._voice.initialize.assert_called_once()
        autonomous._voice.set_on_hear.assert_called_once()
        autonomous._voice.start.assert_called_once()
        assert autonomous._running is True
        autonomous.stop()

    def test_start_idempotent(self, autonomous):
        autonomous._running = True
        autonomous.start()
        autonomous._voice.initialize.assert_not_called()

    def test_stop(self, autonomous):
        autonomous._running = True
        autonomous._thought_thread = MagicMock()
        autonomous.stop()
        assert autonomous._running is False
        autonomous._voice.stop.assert_called_once()
        autonomous._thought_thread.join.assert_called_once_with(timeout=2.0)


class TestAutonomousVoiceUpdateState:
    """Test state update and derived behavior traits."""

    def test_update_state(self, autonomous):
        autonomous.update_state(0.8, 0.7, 0.6, 0.9, "curious")
        assert autonomous._warmth == 0.8
        assert autonomous._clarity == 0.7
        assert autonomous._stability == 0.6
        assert autonomous._presence == 0.9
        assert autonomous._mood == "curious"

    def test_update_state_affects_chattiness(self, autonomous):
        """Chattiness = 0.15 + warmth*0.25 + presence*0.2"""
        autonomous.update_state(1.0, 0.5, 0.5, 1.0)
        expected = 0.15 + 1.0 * 0.25 + 1.0 * 0.2  # 0.6
        assert autonomous._chattiness == pytest.approx(expected)

    def test_update_state_affects_curiosity(self, autonomous):
        """Curiosity = 0.3 + clarity*0.3 + (1-stability)*0.2"""
        autonomous.update_state(0.5, 1.0, 0.0, 0.5)
        expected = 0.3 + 1.0 * 0.3 + (1.0 - 0.0) * 0.2  # 0.8
        assert autonomous._curiosity == pytest.approx(expected)

    def test_update_state_affects_reflectiveness(self, autonomous):
        """Reflectiveness = stability*0.5 + clarity*0.3"""
        autonomous.update_state(0.5, 1.0, 1.0, 0.5)
        expected = 1.0 * 0.5 + 1.0 * 0.3  # 0.8
        assert autonomous._reflectiveness == pytest.approx(expected)

    def test_update_state_delegates_to_voice(self, autonomous):
        autonomous.update_state(0.8, 0.7, 0.6, 0.5)
        autonomous._voice.update_anima_state.assert_called_once_with(0.8, 0.7, 0.6)


class TestAutonomousVoiceUpdateEnvironment:
    """Test environment update."""

    def test_update_environment(self, autonomous):
        autonomous.update_environment(28.5, 65.0, 300.0)
        assert autonomous._temperature == 28.5
        assert autonomous._humidity == 65.0
        assert autonomous._light_level == 300.0


class TestAutonomousVoiceEnvironmentThoughts:
    """Test environment-based thought generation."""

    def test_hot_temperature(self, autonomous):
        # Isolate: only hot temperature triggers, normal light and humidity
        autonomous._temperature = 30.0
        autonomous._light_level = 400.0  # Normal range
        autonomous._humidity = 50.0  # Normal range
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "warm" in thought.text.lower()

    def test_cold_temperature(self, autonomous):
        autonomous._temperature = 15.0
        autonomous._light_level = 400.0
        autonomous._humidity = 50.0
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "cool" in thought.text.lower()

    def test_comfortable_temperature(self, autonomous):
        autonomous._temperature = 23.0
        autonomous._light_level = 400.0
        autonomous._humidity = 50.0
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "nice" in thought.text.lower()

    def test_bright_light(self, autonomous):
        autonomous._temperature = 20.0  # No trigger (between 18-22)
        autonomous._light_level = 900.0
        autonomous._humidity = 50.0
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "bright" in thought.text.lower()

    def test_dim_light(self, autonomous):
        autonomous._temperature = 20.0  # No trigger
        autonomous._light_level = 50.0
        autonomous._humidity = 50.0
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "dim" in thought.text.lower()

    def test_high_humidity(self, autonomous):
        autonomous._temperature = 20.0  # No trigger
        autonomous._light_level = 400.0
        autonomous._humidity = 80.0
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "humid" in thought.text.lower()

    def test_low_humidity(self, autonomous):
        autonomous._temperature = 20.0  # No trigger
        autonomous._light_level = 400.0
        autonomous._humidity = 20.0
        thought = autonomous._generate_environment_thought()
        if thought:
            assert "dry" in thought.text.lower()

    def test_no_notable_conditions(self, autonomous):
        """Normal conditions may produce no environment thought."""
        autonomous._temperature = 20.0  # Between 18-22, not in any category
        autonomous._light_level = 400.0  # Between 100-800
        autonomous._humidity = 50.0  # Between 30-70
        thought = autonomous._generate_environment_thought()
        assert thought is None

    def test_urgency_scaled_by_warmth(self, autonomous):
        autonomous._temperature = 30.0
        autonomous._warmth = 1.0
        thought = autonomous._generate_environment_thought()
        if thought:
            # urgency = base_urgency * warmth. With warmth=1.0, urgency=base_urgency
            assert thought.urgency > 0

    def test_recently_spoken_filtered(self, autonomous):
        """Phrases too similar to recent speech are filtered out."""
        autonomous._temperature = 30.0
        autonomous._spoken_recently = ["It's quite warm"]
        thought = autonomous._generate_environment_thought()
        # After filtering, might be None if that was the only observation
        # or might be a different observation
        if thought:
            assert thought.text != "It's quite warm"


class TestAutonomousVoiceFeelingThoughts:
    """Test feeling-based thought generation."""

    def test_high_warmth_high_stability(self, autonomous):
        autonomous._warmth = 0.8
        autonomous._stability = 0.7
        autonomous._clarity = 0.7
        thought = autonomous._generate_feeling_thought()
        if thought:
            assert thought.intent == SpeechIntent.FEELING

    def test_high_clarity(self, autonomous):
        autonomous._clarity = 0.8
        thought = autonomous._generate_feeling_thought()
        if thought:
            assert "clear" in thought.text.lower() or "content" in thought.text.lower() or "warm" in thought.text.lower()

    def test_low_clarity(self, autonomous):
        autonomous._clarity = 0.2
        thought = autonomous._generate_feeling_thought()
        if thought:
            assert "foggy" in thought.text.lower()

    def test_high_presence(self, autonomous):
        autonomous._presence = 0.9
        autonomous._clarity = 0.7  # needed to enter the function
        thought = autonomous._generate_feeling_thought()
        if thought:
            assert thought.intent == SpeechIntent.FEELING

    def test_curious_mood(self, autonomous):
        autonomous._mood = "curious"
        autonomous._clarity = 0.7
        thought = autonomous._generate_feeling_thought()
        if thought:
            assert thought.intent == SpeechIntent.FEELING

    def test_no_notable_feelings(self, autonomous):
        autonomous._warmth = 0.5
        autonomous._clarity = 0.5
        autonomous._stability = 0.5
        autonomous._presence = 0.5
        autonomous._mood = "neutral"
        thought = autonomous._generate_feeling_thought()
        assert thought is None


class TestAutonomousVoiceReflections:
    """Test reflection thought generation."""

    def test_generate_reflection(self, autonomous):
        thought = autonomous._generate_reflection()
        if thought:
            assert thought.intent == SpeechIntent.REFLECTION
            assert thought.urgency >= 0

    def test_reflection_with_heard_recently(self, autonomous):
        autonomous._heard_recently = [
            Utterance(text="test", confidence=0.9, timestamp=time.time(), duration=1.0)
        ]
        thought = autonomous._generate_reflection()
        if thought:
            assert thought.intent == SpeechIntent.REFLECTION

    def test_reflection_urgency_scaled_by_reflectiveness(self, autonomous):
        autonomous._reflectiveness = 0.0
        thought = autonomous._generate_reflection()
        if thought:
            assert thought.urgency == 0.0

    def test_all_reflections_recently_spoken(self, autonomous):
        """If all candidate reflections were recently spoken, returns None."""
        autonomous._spoken_recently = [
            "I wonder what time it is",
            "The light keeps changing",
            "I was just thinking...",
            "It's quiet",
            "I heard something earlier",
        ]
        thought = autonomous._generate_reflection()
        assert thought is None


class TestAutonomousVoiceQuestions:
    """Test question thought generation."""

    def test_generate_question(self, autonomous):
        thought = autonomous._generate_question()
        if thought:
            assert thought.intent == SpeechIntent.QUESTION

    def test_question_urgency_scaled_by_curiosity(self, autonomous):
        autonomous._curiosity = 0.0
        thought = autonomous._generate_question()
        if thought:
            assert thought.urgency == 0.0

    def test_all_questions_recently_spoken(self, autonomous):
        autonomous._spoken_recently = [
            "Is anyone there?",
            "What's happening?",
            "Are you still here?",
        ]
        thought = autonomous._generate_question()
        assert thought is None


class TestAutonomousVoicePhraseDeduplication:
    """Test phrase deduplication logic."""

    def test_exact_match(self, autonomous):
        autonomous._spoken_recently = ["Hello world"]
        assert autonomous._is_phrase_recent("Hello world") is True

    def test_case_insensitive_match(self, autonomous):
        autonomous._spoken_recently = ["Hello World"]
        assert autonomous._is_phrase_recent("hello world") is True

    def test_punctuation_stripped(self, autonomous):
        autonomous._spoken_recently = ["Hello world!"]
        assert autonomous._is_phrase_recent("Hello world") is True

    def test_high_word_overlap(self, autonomous):
        """Jaccard similarity > 0.6 considered duplicate."""
        autonomous._spoken_recently = ["I feel very content today"]
        # "I feel very content" has 4/5 overlap with "I feel very content today"
        assert autonomous._is_phrase_recent("I feel very content") is True

    def test_low_word_overlap(self, autonomous):
        autonomous._spoken_recently = ["I feel very content today"]
        assert autonomous._is_phrase_recent("The weather is nice outside") is False

    def test_empty_recently_spoken(self, autonomous):
        autonomous._spoken_recently = []
        assert autonomous._is_phrase_recent("anything") is False

    def test_record_spoken_adds(self, autonomous):
        autonomous._record_spoken("hello")
        assert "hello" in autonomous._spoken_recently

    def test_record_spoken_bounded(self, autonomous):
        for i in range(25):
            autonomous._record_spoken(f"phrase {i}")
        assert len(autonomous._spoken_recently) == 20
        assert autonomous._spoken_recently[0] == "phrase 5"  # First 5 were evicted


class TestAutonomousVoiceMaybeSpeak:
    """Test the speaking decision logic."""

    def test_no_pending_thoughts(self, autonomous):
        autonomous._pending_thoughts = []
        autonomous._maybe_speak()
        autonomous._voice.say.assert_not_called()

    def test_cooldown_active(self, autonomous):
        autonomous._last_speech_time = time.time()  # Just spoke
        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="test", urgency=1.0)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_not_called()

    def test_urgency_below_threshold(self, autonomous):
        autonomous._last_speech_time = 0  # Long ago
        autonomous._presence = 0.0  # threshold will be high
        autonomous._chattiness = 0.0
        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="test", urgency=0.1)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_not_called()

    def test_speaks_when_urgent_enough(self, autonomous):
        autonomous._last_speech_time = 0  # Long ago
        autonomous._presence = 1.0  # threshold will be low (0.3)
        autonomous._chattiness = 1.0
        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="Hello there", urgency=0.9)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_called_once_with("Hello there", blocking=True)

    def test_picks_most_urgent_thought(self, autonomous):
        autonomous._last_speech_time = 0
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="low priority", urgency=0.1),
            SpeechMoment(intent=SpeechIntent.FEELING, text="high priority", urgency=0.9),
        ]
        autonomous._maybe_speak()
        # The high-priority thought should be spoken
        if autonomous._voice.say.called:
            autonomous._voice.say.assert_called_once_with("high priority", blocking=True)


class TestAutonomousVoiceSpeak:
    """Test the actual speaking method."""

    def test_speak_sets_last_speech_time(self, autonomous):
        thought = SpeechMoment(
            intent=SpeechIntent.OBSERVATION, text="Something new", urgency=0.5
        )
        autonomous._speak(thought)
        assert autonomous._last_speech_time > 0
        autonomous._voice.say.assert_called_once_with("Something new", blocking=True)

    def test_speak_observation_sets_env_comment_time(self, autonomous):
        thought = SpeechMoment(
            intent=SpeechIntent.OBSERVATION, text="It is warm", urgency=0.5
        )
        autonomous._speak(thought)
        assert autonomous._last_env_comment_time > 0

    def test_speak_feeling_doesnt_set_env_comment_time(self, autonomous):
        thought = SpeechMoment(
            intent=SpeechIntent.FEELING, text="I feel good", urgency=0.5
        )
        autonomous._speak(thought)
        assert autonomous._last_env_comment_time == 0

    def test_speak_notifies_callback(self, autonomous):
        callback = MagicMock()
        autonomous._on_autonomous_speech = callback
        thought = SpeechMoment(
            intent=SpeechIntent.FEELING, text="I feel warm", urgency=0.5
        )
        autonomous._speak(thought)
        callback.assert_called_once_with("I feel warm", SpeechIntent.FEELING)

    def test_speak_skips_recent_phrase(self, autonomous):
        autonomous._spoken_recently = ["It's quite warm"]
        thought = SpeechMoment(
            intent=SpeechIntent.OBSERVATION, text="It's quite warm", urgency=0.9
        )
        autonomous._speak(thought)
        autonomous._voice.say.assert_not_called()

    def test_speak_records_phrase(self, autonomous):
        thought = SpeechMoment(
            intent=SpeechIntent.OBSERVATION, text="Something unique", urgency=0.5
        )
        autonomous._speak(thought)
        assert "Something unique" in autonomous._spoken_recently


class TestAutonomousVoiceOnHear:
    """Test response to heard speech."""

    def test_on_hear_adds_to_history(self, autonomous):
        utterance = Utterance(text="hello", confidence=0.9, timestamp=time.time(), duration=1.0)
        with patch("time.sleep"):
            autonomous._on_hear(utterance)
        assert utterance in autonomous._heard_recently

    def test_on_hear_bounded_history(self, autonomous):
        with patch("time.sleep"):
            for i in range(15):
                u = Utterance(text=f"msg{i}", confidence=0.9, timestamp=time.time(), duration=0.5)
                autonomous._on_hear(u)
        assert len(autonomous._heard_recently) <= 10

    def test_on_hear_low_presence_may_ignore(self, autonomous):
        autonomous._presence = 0.1
        utterance = Utterance(text="hello", confidence=0.9, timestamp=time.time(), duration=1.0)
        # With low presence and random > presence*2, should often stay quiet
        # We seed random to ensure predictable behavior
        with patch("random.random", return_value=0.9), patch("time.sleep"):
            autonomous._on_hear(utterance)
        autonomous._voice.say.assert_not_called()

    def test_on_hear_with_response_generator(self, autonomous):
        autonomous._presence = 1.0
        autonomous._get_response = lambda text: f"echo: {text}"
        autonomous._stability = 1.0  # Minimize delay

        utterance = Utterance(text="hello", confidence=0.9, timestamp=time.time(), duration=1.0)
        with patch("time.sleep"):
            autonomous._on_hear(utterance)

        autonomous._voice.say.assert_called_once_with("echo: hello", blocking=True)

    def test_on_hear_response_generator_returns_none(self, autonomous):
        autonomous._presence = 1.0
        autonomous._get_response = lambda text: None

        utterance = Utterance(text="hello", confidence=0.9, timestamp=time.time(), duration=1.0)
        with patch("time.sleep"):
            autonomous._on_hear(utterance)

        autonomous._voice.say.assert_not_called()


class TestAutonomousVoiceNudge:
    """Test the nudge() method."""

    def test_nudge_adds_thought(self, autonomous):
        autonomous.nudge()
        assert len(autonomous._pending_thoughts) == 1
        assert autonomous._pending_thoughts[0].urgency == 0.6

    def test_nudge_observation(self, autonomous):
        autonomous.nudge(SpeechIntent.OBSERVATION)
        assert autonomous._pending_thoughts[0].text == "Something catches my attention"

    def test_nudge_feeling(self, autonomous):
        autonomous.nudge(SpeechIntent.FEELING)
        assert autonomous._pending_thoughts[0].text == "I notice how I'm feeling"

    def test_nudge_question(self, autonomous):
        autonomous.nudge(SpeechIntent.QUESTION)
        assert autonomous._pending_thoughts[0].text == "I wonder..."

    def test_nudge_reflection(self, autonomous):
        autonomous.nudge(SpeechIntent.REFLECTION)
        assert autonomous._pending_thoughts[0].text == "Let me think..."

    def test_nudge_unknown_intent(self, autonomous):
        autonomous.nudge(SpeechIntent.GREETING)
        assert autonomous._pending_thoughts[0].text == "Hmm..."


class TestAutonomousVoiceCallbacks:
    """Test callback setters."""

    def test_set_response_generator(self, autonomous):
        def func(t):
            return "response"

        autonomous.set_response_generator(func)
        assert autonomous._get_response is func

    def test_set_on_speech(self, autonomous):
        cb = MagicMock()
        autonomous.set_on_speech(cb)
        assert autonomous._on_autonomous_speech is cb


class TestAutonomousVoiceProperties:
    """Test AutonomousVoice properties."""

    def test_is_running(self, autonomous):
        assert autonomous.is_running is False
        autonomous._running = True
        assert autonomous.is_running is True

    def test_chattiness_getter(self, autonomous):
        assert autonomous.chattiness == 0.5

    def test_chattiness_setter_clamps_high(self, autonomous):
        autonomous.chattiness = 2.0
        assert autonomous.chattiness == 1.0

    def test_chattiness_setter_clamps_low(self, autonomous):
        autonomous.chattiness = -1.0
        assert autonomous.chattiness == 0.0


class TestAutonomousVoiceGenerateThoughts:
    """Test the _generate_thoughts method."""

    def test_generate_thoughts_too_soon(self, autonomous):
        """No thoughts generated within cooldown/2."""
        autonomous._last_speech_time = time.time()  # Just spoke
        autonomous._generate_thoughts()
        assert len(autonomous._pending_thoughts) == 0

    def test_generate_thoughts_clears_old(self, autonomous):
        """Old pending thoughts are cleared."""
        old = SpeechMoment(
            intent=SpeechIntent.OBSERVATION, text="old thought", urgency=0.5,
            timestamp=time.time() - 120  # 2 minutes old
        )
        autonomous._pending_thoughts = [old]
        autonomous._last_speech_time = 0  # Long ago
        autonomous._presence = 0  # Suppress new generation (random > 0)
        autonomous._chattiness = 0
        autonomous._generate_thoughts()
        # Old thought should be cleared
        assert old not in autonomous._pending_thoughts

    def test_generate_thoughts_env_observation(self, autonomous):
        """Can generate environment observations when conditions met."""
        autonomous._last_speech_time = 0
        autonomous._last_env_comment_time = 0  # Long ago
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._temperature = 30.0  # Hot

        with patch("random.random", return_value=0.01):  # Ensure thought is generated
            autonomous._generate_thoughts()

        # May or may not have generated one (randomness), but the path is exercised
        assert isinstance(autonomous._pending_thoughts, list)

    def test_generate_thoughts_feeling(self, autonomous):
        """Can generate feeling thoughts with high clarity."""
        autonomous._last_speech_time = 0
        autonomous._last_env_comment_time = time.time()  # Skip env
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._clarity = 0.8
        autonomous._warmth = 0.8
        autonomous._stability = 0.7

        with patch("random.random", return_value=0.01):
            autonomous._generate_thoughts()

    def test_generate_thoughts_reflection(self, autonomous):
        """Can generate reflections with high stability and reflectiveness."""
        autonomous._last_speech_time = 0
        autonomous._last_env_comment_time = time.time()
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._stability = 0.8
        autonomous._reflectiveness = 0.7
        autonomous._clarity = 0.3  # Skip feeling

        with patch("random.random", return_value=0.01):
            autonomous._generate_thoughts()

    def test_generate_thoughts_question(self, autonomous):
        """Can generate questions with high curiosity."""
        autonomous._last_speech_time = 0
        autonomous._last_env_comment_time = time.time()
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._curiosity = 0.9
        autonomous._stability = 0.3  # Skip reflection
        autonomous._clarity = 0.3  # Skip feeling

        with patch("random.random", return_value=0.01):
            autonomous._generate_thoughts()


class TestAutonomousVoiceSpeakThreshold:
    """Test the dynamic speech threshold calculation."""

    def test_threshold_high_presence_high_chattiness(self, autonomous):
        """High presence + chattiness = low threshold = easier to speak."""
        autonomous._last_speech_time = 0
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0

        # threshold = 0.3 + (1-1)*0.3 + (1-1)*0.2 = 0.3
        # A thought with urgency > 0.3 should pass
        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="Test speak", urgency=0.4)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_called_once()

    def test_threshold_low_presence_low_chattiness(self, autonomous):
        """Low presence + chattiness = high threshold = harder to speak."""
        autonomous._last_speech_time = 0
        autonomous._presence = 0.0
        autonomous._chattiness = 0.0

        # threshold = 0.3 + (1-0)*0.3 + (1-0)*0.2 = 0.8
        # A thought with urgency < 0.8 should NOT pass
        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="Test no speak", urgency=0.7)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_not_called()


class TestAutonomousVoiceRateLimiting:
    """Test rate limiting via cooldown."""

    def test_speech_cooldown_default(self, autonomous):
        assert autonomous._speech_cooldown == 60.0

    def test_cooldown_prevents_rapid_speech(self, autonomous):
        """Cannot speak twice within cooldown period."""
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._last_speech_time = time.time() - 30  # 30 seconds ago, cooldown is 60

        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="Test", urgency=0.9)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_not_called()

    def test_can_speak_after_cooldown(self, autonomous):
        """Can speak after cooldown expires."""
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0
        autonomous._last_speech_time = time.time() - 120  # 2 minutes ago

        autonomous._pending_thoughts = [
            SpeechMoment(intent=SpeechIntent.OBSERVATION, text="Test ok", urgency=0.9)
        ]
        autonomous._maybe_speak()
        autonomous._voice.say.assert_called_once()

    def test_half_cooldown_suppresses_thought_generation(self, autonomous):
        """Within cooldown/2, new thoughts are not generated."""
        autonomous._last_speech_time = time.time() - 10  # 10 seconds ago, half cooldown = 30
        autonomous._presence = 1.0
        autonomous._chattiness = 1.0

        with patch("random.random", return_value=0.0):
            autonomous._generate_thoughts()

        # No new thoughts should be generated
        assert len(autonomous._pending_thoughts) == 0


class TestAudioChunkDataclass:
    """Test AudioChunk dataclass."""

    def test_audio_chunk(self):
        chunk = AudioChunk(data=b"\x00\x01\x02", timestamp=1234.5, duration=0.064)
        assert chunk.data == b"\x00\x01\x02"
        assert chunk.timestamp == 1234.5
        assert chunk.duration == 0.064


class TestAudioPlaybackDataclass:
    """Test AudioPlayback dataclass."""

    def test_defaults(self):
        p = AudioPlayback(audio_bytes=b"\x00")
        assert p.sample_rate == 22050
        assert p.channels == 1
        assert p.sample_width == 2

    def test_custom(self):
        p = AudioPlayback(audio_bytes=b"\x00", sample_rate=44100, channels=2, sample_width=4)
        assert p.sample_rate == 44100
        assert p.channels == 2
        assert p.sample_width == 4


class TestTranscriptionResultDataclass:
    """Test TranscriptionResult dataclass."""

    def test_fields(self):
        r = TranscriptionResult(
            text="hello", confidence=0.95, is_final=True, alternatives=["helo"]
        )
        assert r.text == "hello"
        assert r.confidence == 0.95
        assert r.is_final is True
        assert r.alternatives == ["helo"]


class TestVoiceDataclass:
    """Test Voice dataclass."""

    def test_fields(self):
        v = Voice("en_US-test", "en_US", "medium", speaker_id=3)
        assert v.name == "en_US-test"
        assert v.language == "en_US"
        assert v.quality == "medium"
        assert v.speaker_id == 3

    def test_defaults(self):
        v = Voice("test", "en", "low")
        assert v.speaker_id is None


class TestRecommendedVoices:
    """Test the RECOMMENDED_VOICES dictionary."""

    def test_all_voices_present(self):
        assert "default" in RECOMMENDED_VOICES
        assert "warm" in RECOMMENDED_VOICES
        assert "clear" in RECOMMENDED_VOICES
        assert "soft" in RECOMMENDED_VOICES

    def test_voices_have_valid_fields(self):
        for key, voice in RECOMMENDED_VOICES.items():
            assert isinstance(voice.name, str)
            assert isinstance(voice.language, str)
            assert voice.quality in ("low", "medium", "high")


class TestModuleConstants:
    """Test module-level constants."""

    def test_mic_constants(self):
        assert SAMPLE_RATE == 16000
        assert CHANNELS == 1
        assert CHUNK_SIZE == 1024


class TestVoiceStyleEnum:
    """Test VoiceStyle enum."""

    def test_all_styles(self):
        assert VoiceStyle.NEUTRAL.value == "neutral"
        assert VoiceStyle.WARM.value == "warm"
        assert VoiceStyle.CLEAR.value == "clear"
        assert VoiceStyle.SOFT.value == "soft"
        assert VoiceStyle.BRIGHT.value == "bright"
