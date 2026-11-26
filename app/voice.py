"""Pluggable Genuine STT & TTS Voice Provider Registry (Cloud + Local Faster-Whisper / Piper / System Speech + Browser Fallback)."""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import get_runtime_gemini_key, get_runtime_secret, settings

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    transcript: str
    provider: str
    latency_ms: int
    status: str = "ok"  # "ok" | "fallback" | "error"
    language_hint: Optional[str] = None
    timestamps: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    error: Optional[str] = None


@dataclass
class TTSResult:
    audio_bytes: bytes
    content_type: str
    provider: str
    latency_ms: int
    voice: str = "default"
    status: str = "ok"  # "ok" | "fallback" | "error"
    language: str = "en-US"
    fallback_used: bool = False
    error: Optional[str] = None


def cleanup_temp_audio(ttl_minutes: Optional[int] = None) -> int:
    """Remove expired temporary audio files older than TTL."""
    tmp_dir = Path(settings.temp_audio_dir)
    if not tmp_dir.exists():
        return 0
    ttl_sec = (ttl_minutes if ttl_minutes is not None else settings.temp_audio_ttl_minutes) * 60
    cutoff = time.time() - ttl_sec
    removed = 0
    for f in tmp_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


class STTProvider(ABC):
    name: str = "base"
    kind: str = "cloud"

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        ...


class TTSProvider(ABC):
    name: str = "base"
    kind: str = "cloud"

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def active_voice(self, locale: str = "en-US") -> str:
        ...

    @abstractmethod
    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        ...


# ---------------- Cloud & Local STT Providers ----------------

class DeepgramSTTProvider(STTProvider):
    name = "deepgram"
    kind = "cloud"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("DEEPGRAM_API_KEY", settings.deepgram_api_key))

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("DEEPGRAM_API_KEY", settings.deepgram_api_key)
        params: dict[str, str] = {"model": "nova-2", "smart_format": "true", "utterances": "true"}
        if language and language != "hinglish":
            params["language"] = language
        else:
            params["detect_language"] = "true"

        with httpx.Client(timeout=12.0) as client:
            resp = client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers={"Authorization": f"Token {api_key}", "Content-Type": mime_type},
                content=audio_bytes,
            )
            resp.raise_for_status()
            data = resp.json()
            alt = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
            transcript = (alt.get("transcript") or "").strip()
            words = alt.get("words") or []
            timestamps = [
                {"word": w.get("word", ""), "start": w.get("start", 0.0), "end": w.get("end", 0.0)}
                for w in words[:50]
            ]
            return STTResult(
                transcript=transcript,
                provider=self.name,
                status="ok",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                language_hint=language,
                timestamps=timestamps,
            )


class WhisperSTTProvider(STTProvider):
    name = "whisper"
    kind = "cloud"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("OPENAI_API_KEY", settings.openai_api_key))

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("OPENAI_API_KEY", settings.openai_api_key)
        ext = "wav" if "wav" in mime_type else "webm"
        files = {"file": (f"audio.{ext}", audio_bytes, mime_type)}
        data: dict[str, str] = {"model": "whisper-1", "response_format": "verbose_json"}
        if language and language != "hinglish":
            data["language"] = language

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
            )
            resp.raise_for_status()
            body = resp.json()
            transcript = (body.get("text") or "").strip()
            segs = [
                {"text": s.get("text", ""), "start": s.get("start", 0.0), "end": s.get("end", 0.0)}
                for s in (body.get("segments") or [])
            ]
            return STTResult(
                transcript=transcript,
                provider=self.name,
                status="ok",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                language_hint=language,
                timestamps=segs,
            )


class GeminiSTTProvider(STTProvider):
    name = "gemini"
    kind = "cloud"

    def is_configured(self) -> bool:
        return bool(get_runtime_gemini_key())

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=get_runtime_gemini_key())
        prompt = "Transcribe this spoken sales prospect audio verbatim (supports English, Hindi, Hinglish, ES, FR, DE). Return only the spoken text."
        if language:
            prompt += f" Language hint: {language}."
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        return STTResult(
            transcript=(resp.text or "").strip(),
            provider=self.name,
            status="ok",
            latency_ms=int((time.perf_counter() - t0) * 1000),
            language_hint=language,
        )


class FasterWhisperSTTProvider(STTProvider):
    """Genuine offline speech-to-text provider using faster-whisper with segment timestamps."""

    name = "faster-whisper"
    kind = "local"

    def __init__(self) -> None:
        self._model = None

    def is_configured(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        if not self.is_configured():
            raise RuntimeError("faster-whisper package is not installed.")
        tmp_dir = Path(settings.temp_audio_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp_audio()
        ext = ".wav" if "wav" in mime_type else ".webm"
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=ext, delete=False) as tf:
            tf.write(audio_bytes)
            tmp_path = Path(tf.name)
        try:
            model = self._load_model()
            lang_arg = language if (language and language != "hinglish") else None
            segments_iter, info = model.transcribe(str(tmp_path), language=lang_arg)
            segments_list = list(segments_iter)
            transcript = " ".join(seg.text.strip() for seg in segments_list if seg.text.strip()).strip()
            timestamps = [
                {"text": seg.text.strip(), "start": round(seg.start, 2), "end": round(seg.end, 2)}
                for seg in segments_list
            ]
            return STTResult(
                transcript=transcript,
                provider=self.name,
                status="ok",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                language_hint=getattr(info, "language", language),
                timestamps=timestamps,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


class BrowserFallbackSTTProvider(STTProvider):
    name = "browser-webspeech"
    kind = "browser-fallback"

    def is_configured(self) -> bool:
        return True

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        return STTResult(
            transcript="",
            provider=self.name,
            status="fallback",
            latency_ms=0,
            language_hint=language,
            fallback_used=True,
            error="No server STT engine available; browser Web Speech API handles live speech recognition.",
        )


# ---------------- Cloud & Genuine Offline TTS Providers ----------------

class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"
    kind = "cloud"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("ELEVENLABS_API_KEY", settings.elevenlabs_api_key))

    def active_voice(self, locale: str = "en-US") -> str:
        return get_runtime_secret("ELEVENLABS_VOICE_ID", settings.elevenlabs_voice_id)

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("ELEVENLABS_API_KEY", settings.elevenlabs_api_key)
        voice_id = self.active_voice(locale)
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            )
            resp.raise_for_status()
            return TTSResult(
                audio_bytes=resp.content,
                content_type="audio/mpeg",
                provider=self.name,
                voice=voice_id,
                status="ok",
                language=locale,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )


class GoogleCloudTTSProvider(TTSProvider):
    name = "google-cloud-tts"
    kind = "cloud"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("GOOGLE_TTS_API_KEY", settings.google_tts_api_key))

    def active_voice(self, locale: str = "en-US") -> str:
        return f"{locale}-Neural2"

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("GOOGLE_TTS_API_KEY", settings.google_tts_api_key)
        with httpx.Client(timeout=12.0) as client:
            resp = client.post(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}",
                json={
                    "input": {"text": text},
                    "voice": {"languageCode": locale},
                    "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate},
                },
            )
            resp.raise_for_status()
            b64 = resp.json().get("audioContent", "")
            return TTSResult(
                audio_bytes=base64.b64decode(b64),
                content_type="audio/mpeg",
                provider=self.name,
                voice=self.active_voice(locale),
                status="ok",
                language=locale,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )


class PiperTTSProvider(TTSProvider):
    """Genuine offline neural TTS provider via Piper (.onnx voice models for EN / HI)."""

    name = "piper"
    kind = "local"

    def _resolve_model_path(self, locale: str = "en-US") -> Optional[Path]:
        is_hindi = locale.lower().startswith("hi")
        configured = settings.piper_model_hi if is_hindi else settings.piper_model_en
        if configured and Path(configured).exists():
            return Path(configured)
        voices_dir = Path(settings.piper_voices_dir)
        if voices_dir.exists():
            pattern = "*hi*.onnx" if is_hindi else "*en*.onnx"
            matches = sorted(voices_dir.glob(pattern))
            if matches:
                return matches[0]
            any_onnx = sorted(voices_dir.glob("*.onnx"))
            if any_onnx:
                return any_onnx[0]
        return None

    def is_configured(self) -> bool:
        has_pkg = importlib.util.find_spec("piper") is not None
        has_bin = shutil.which("piper") is not None
        return (has_pkg or has_bin) and (self._resolve_model_path("en-US") is not None or self._resolve_model_path("hi-IN") is not None)

    def active_voice(self, locale: str = "en-US") -> str:
        model = self._resolve_model_path(locale)
        return model.stem if model else "unconfigured-onnx"

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        t0 = time.perf_counter()
        model_path = self._resolve_model_path(locale)
        if not model_path:
            raise FileNotFoundError(f"No Piper ONNX voice model found for locale {locale} in {settings.piper_voices_dir}.")

        if importlib.util.find_spec("piper") is not None:
            from piper.voice import PiperVoice

            voice = PiperVoice.load(str(model_path))
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize(text, wf)
            return TTSResult(
                audio_bytes=buf.getvalue(),
                content_type="audio/wav",
                provider=self.name,
                voice=model_path.stem,
                status="ok",
                language=locale,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )

        piper_bin = shutil.which("piper")
        if piper_bin:
            tmp_dir = Path(settings.temp_audio_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".wav", delete=False) as tf:
                out_wav = Path(tf.name)
            try:
                subprocess.run(
                    [piper_bin, "--model", str(model_path), "--output_file", str(out_wav)],
                    input=text.encode("utf-8"),
                    check=True,
                    capture_output=True,
                    timeout=15,
                )
                return TTSResult(
                    audio_bytes=out_wav.read_bytes(),
                    content_type="audio/wav",
                    provider=self.name,
                    voice=model_path.stem,
                    status="ok",
                    language=locale,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
            finally:
                out_wav.unlink(missing_ok=True)

        raise RuntimeError("Piper TTS library/binary not available.")


class WindowsSapiTTSProvider(TTSProvider):
    """Genuine offline speech synthesizer using Windows .NET System.Speech (zero external keys/downloads)."""

    name = "windows-sapi"
    kind = "local"

    def is_configured(self) -> bool:
        return platform.system().lower() == "windows"

    def active_voice(self, locale: str = "en-US") -> str:
        return "System.Speech.SpeechSynthesizer"

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        if not self.is_configured():
            raise RuntimeError("Windows SAPI is only available on Windows hosts.")
        t0 = time.perf_counter()
        tmp_dir = Path(settings.temp_audio_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp_audio()

        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".wav", delete=False) as tf:
            out_path = Path(tf.name)

        sapi_rate = max(-5, min(5, int(round((rate - 1.0) * 5))))
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {sapi_rate}; "
            f"$synth.SetOutputToWaveFile('{str(out_path).replace(chr(39), chr(39)*2)}'); "
            "$txt = [Console]::In.ReadToEnd(); "
            "$synth.Speak($txt); "
            "$synth.Dispose();"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=12,
            )
            audio_bytes = out_path.read_bytes()
            if len(audio_bytes) <= 44 or audio_bytes[:4] != b"RIFF":
                raise RuntimeError("Windows SAPI produced empty or invalid WAV output.")
            return TTSResult(
                audio_bytes=audio_bytes,
                content_type="audio/wav",
                provider=self.name,
                voice=self.active_voice(locale),
                status="ok",
                language=locale,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                fallback_used=False,
            )
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass


class BrowserFallbackTTSProvider(TTSProvider):
    """Honest fallback when no server-side speech synthesizer is available (never emits fake tones)."""

    name = "browser-webspeech"
    kind = "browser-fallback"

    def is_configured(self) -> bool:
        return True

    def active_voice(self, locale: str = "en-US") -> str:
        return f"window.speechSynthesis ({locale})"

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        return TTSResult(
            audio_bytes=b"",
            content_type="application/json",
            provider=self.name,
            voice=self.active_voice(locale),
            status="fallback",
            language=locale,
            latency_ms=0,
            fallback_used=True,
            error="No server speech model configured; browser SpeechSynthesis handles audio playback.",
        )


class VoiceProviderRegistry:
    """Routes STT and TTS calls to configured cloud or genuine local providers with honest fallbacks."""

    def __init__(self) -> None:
        self.stt_providers: dict[str, STTProvider] = {
            "deepgram": DeepgramSTTProvider(),
            "whisper": WhisperSTTProvider(),
            "gemini": GeminiSTTProvider(),
            "faster-whisper": FasterWhisperSTTProvider(),
            "browser": BrowserFallbackSTTProvider(),
        }
        self.tts_providers: dict[str, TTSProvider] = {
            "elevenlabs": ElevenLabsTTSProvider(),
            "google": GoogleCloudTTSProvider(),
            "piper": PiperTTSProvider(),
            "windows-sapi": WindowsSapiTTSProvider(),
            "browser": BrowserFallbackTTSProvider(),
        }

    def active_stt_name(self) -> str:
        pref = get_runtime_secret("STT_PROVIDER", settings.stt_provider).lower()
        if pref in self.stt_providers and self.stt_providers[pref].is_configured():
            return self.stt_providers[pref].name
        for key in ("deepgram", "whisper", "gemini", "faster-whisper"):
            if self.stt_providers[key].is_configured():
                return self.stt_providers[key].name
        return "browser-webspeech"

    def active_tts_name(self) -> str:
        pref = get_runtime_secret("TTS_PROVIDER", settings.tts_provider).lower()
        if pref in self.tts_providers and self.tts_providers[pref].is_configured():
            return self.tts_providers[pref].name
        for key in ("elevenlabs", "google", "piper", "windows-sapi"):
            if self.tts_providers[key].is_configured():
                return self.tts_providers[key].name
        return "browser-webspeech"

    def provider_status(self) -> dict[str, Any]:
        """Expose active STT/TTS providers and availability without leaking credentials."""
        active_tts = self.active_tts_name()
        tts_obj = next((p for p in self.tts_providers.values() if p.name == active_tts), self.tts_providers["browser"])
        return {
            "active_stt": self.active_stt_name(),
            "active_tts": active_tts,
            "active_tts_voice": tts_obj.active_voice(),
            "stt_available": {
                k: {"name": p.name, "kind": p.kind, "configured": p.is_configured()}
                for k, p in self.stt_providers.items()
            },
            "tts_available": {
                k: {"name": p.name, "kind": p.kind, "configured": p.is_configured()}
                for k, p in self.tts_providers.items()
            },
        }

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        if not audio_bytes:
            return STTResult(
                transcript="",
                provider="none",
                status="error",
                latency_ms=0,
                fallback_used=True,
                error="Empty audio payload.",
            )
        pref = get_runtime_secret("STT_PROVIDER", settings.stt_provider).lower()
        order = [pref] if pref in self.stt_providers else ["deepgram", "whisper", "gemini", "faster-whisper", "browser"]
        if "browser" not in order:
            order.append("browser")

        last_err: Optional[str] = None
        for key in order:
            prov = self.stt_providers[key]
            if not prov.is_configured():
                continue
            try:
                res = prov.transcribe(audio_bytes, mime_type=mime_type, language=language)
                if res.transcript or key == "browser":
                    if last_err and key == "browser":
                        res.error = last_err
                    return res
            except Exception as exc:
                last_err = f"STT provider {key} failed: {exc}"
                logger.warning(last_err)
        fb = self.stt_providers["browser"].transcribe(audio_bytes, mime_type=mime_type, language=language)
        if last_err:
            fb.error = last_err
        return fb

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        if not text or not text.strip():
            return TTSResult(
                audio_bytes=b"",
                content_type="application/json",
                provider="none",
                voice="none",
                status="error",
                language=locale,
                latency_ms=0,
                fallback_used=True,
                error="Cannot synthesize empty text.",
            )
        pref = get_runtime_secret("TTS_PROVIDER", settings.tts_provider).lower()
        order = [pref] if pref in self.tts_providers else ["elevenlabs", "google", "piper", "windows-sapi", "browser"]
        if "browser" not in order:
            order.append("browser")

        last_err: Optional[str] = None
        for key in order:
            prov = self.tts_providers[key]
            if not prov.is_configured():
                continue
            try:
                return prov.synthesize(text=text, locale=locale, rate=rate)
            except Exception as exc:
                last_err = f"TTS provider {key} failed: {exc}"
                logger.warning(last_err)
        fb = self.tts_providers["browser"].synthesize(text=text, locale=locale, rate=rate)
        if last_err:
            fb.error = last_err
        return fb
