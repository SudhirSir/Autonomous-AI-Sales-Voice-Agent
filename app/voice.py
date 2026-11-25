"""Pluggable STT and TTS Voice Provider Registry with timeouts and graceful fallback."""

from __future__ import annotations

import base64
import io
import logging
import math
import struct
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import get_runtime_gemini_key, get_runtime_secret, settings

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    transcript: str
    provider: str
    latency_ms: int
    language_hint: Optional[str] = None
    fallback_used: bool = False
    error: Optional[str] = None


@dataclass
class TTSResult:
    audio_bytes: bytes
    content_type: str
    provider: str
    latency_ms: int
    fallback_used: bool = False
    error: Optional[str] = None


class STTProvider(ABC):
    name: str = "base"

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

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        ...


class DeepgramSTTProvider(STTProvider):
    name = "deepgram"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("DEEPGRAM_API_KEY", settings.deepgram_api_key))

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("DEEPGRAM_API_KEY", settings.deepgram_api_key)
        params = {"model": "nova-2", "smart_format": "true"}
        if language and language != "hinglish":
            params["language"] = language
        elif language == "hinglish":
            params["detect_language"] = "true"
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
            return STTResult(
                transcript=transcript,
                provider=self.name,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                language_hint=language,
            )


class WhisperSTTProvider(STTProvider):
    name = "whisper"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("OPENAI_API_KEY", settings.openai_api_key))

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("OPENAI_API_KEY", settings.openai_api_key)
        ext = "wav" if "wav" in mime_type else "webm"
        files = {"file": (f"audio.{ext}", audio_bytes, mime_type)}
        data = {"model": "whisper-1"}
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
            transcript = (resp.json().get("text") or "").strip()
            return STTResult(
                transcript=transcript,
                provider=self.name,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                language_hint=language,
            )


class GeminiSTTProvider(STTProvider):
    name = "gemini"

    def is_configured(self) -> bool:
        return bool(get_runtime_gemini_key())

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        t0 = time.perf_counter()
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=get_runtime_gemini_key())
        prompt = "Transcribe this spoken sales prospect audio verbatim. Return only the spoken text."
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
            latency_ms=int((time.perf_counter() - t0) * 1000),
            language_hint=language,
        )


class BrowserFallbackSTTProvider(STTProvider):
    name = "browser-webspeech"

    def is_configured(self) -> bool:
        return True

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        return STTResult(
            transcript="",
            provider=self.name,
            latency_ms=0,
            language_hint=language,
            fallback_used=True,
            error="No external server STT API key configured; browser Web Speech API handles real-time STT.",
        )


class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("ELEVENLABS_API_KEY", settings.elevenlabs_api_key))

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        t0 = time.perf_counter()
        api_key = get_runtime_secret("ELEVENLABS_API_KEY", settings.elevenlabs_api_key)
        voice_id = get_runtime_secret("ELEVENLABS_VOICE_ID", settings.elevenlabs_voice_id)
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
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )


class GoogleCloudTTSProvider(TTSProvider):
    name = "google-cloud-tts"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("GOOGLE_TTS_API_KEY", settings.google_tts_api_key))

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
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )


class LocalWavTTSProvider(TTSProvider):
    """Generates a deterministic, valid PCM WAV audio buffer as a server-side fallback."""

    name = "local-wav"

    def is_configured(self) -> bool:
        return True

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        t0 = time.perf_counter()
        sample_rate = 16000
        # Modulate gentle dual-tone speech-cadence envelope proportional to utterance length
        duration_sec = min(1.8, max(0.35, len(text) * 0.012 / max(0.5, rate)))
        num_samples = int(sample_rate * duration_sec)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for i in range(num_samples):
                t = i / sample_rate
                env = math.sin(math.pi * min(1.0, t / duration_sec))
                freq = 320.0 + 45.0 * math.sin(2.0 * math.pi * 4.0 * t)
                val = int(4200 * env * math.sin(2.0 * math.pi * freq * t))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(bytes(frames))

        return TTSResult(
            audio_bytes=buf.getvalue(),
            content_type="audio/wav",
            provider=self.name,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            fallback_used=True,
        )


class VoiceProviderRegistry:
    """Routes STT and TTS calls to configured providers with safe fallbacks."""

    def __init__(self) -> None:
        self.stt_providers: dict[str, STTProvider] = {
            "deepgram": DeepgramSTTProvider(),
            "whisper": WhisperSTTProvider(),
            "gemini": GeminiSTTProvider(),
            "browser": BrowserFallbackSTTProvider(),
        }
        self.tts_providers: dict[str, TTSProvider] = {
            "elevenlabs": ElevenLabsTTSProvider(),
            "google": GoogleCloudTTSProvider(),
            "local-wav": LocalWavTTSProvider(),
        }

    def active_stt_name(self) -> str:
        pref = get_runtime_secret("STT_PROVIDER", settings.stt_provider).lower()
        if pref in self.stt_providers and self.stt_providers[pref].is_configured():
            return self.stt_providers[pref].name
        for key in ("deepgram", "whisper", "gemini"):
            if self.stt_providers[key].is_configured():
                return self.stt_providers[key].name
        return "browser-webspeech"

    def active_tts_name(self) -> str:
        pref = get_runtime_secret("TTS_PROVIDER", settings.tts_provider).lower()
        if pref == "browser":
            return "browser-webspeech"
        if pref in self.tts_providers and self.tts_providers[pref].is_configured():
            return self.tts_providers[pref].name
        for key in ("elevenlabs", "google"):
            if self.tts_providers[key].is_configured():
                return self.tts_providers[key].name
        return "browser-webspeech+local-wav"

    def transcribe(
        self, audio_bytes: bytes, mime_type: str = "audio/webm", language: Optional[str] = None
    ) -> STTResult:
        if not audio_bytes:
            return STTResult(
                transcript="",
                provider="none",
                latency_ms=0,
                fallback_used=True,
                error="Empty audio payload.",
            )
        pref = get_runtime_secret("STT_PROVIDER", settings.stt_provider).lower()
        order = [pref] if pref in self.stt_providers else ["deepgram", "whisper", "gemini", "browser"]
        if "browser" not in order:
            order.append("browser")

        for key in order:
            prov = self.stt_providers[key]
            if not prov.is_configured():
                continue
            try:
                res = prov.transcribe(audio_bytes, mime_type=mime_type, language=language)
                if res.transcript or key == "browser":
                    return res
            except Exception as exc:
                logger.warning("STT provider %s failed, falling back: %s", key, exc)
        return self.stt_providers["browser"].transcribe(audio_bytes, mime_type=mime_type, language=language)

    def synthesize(self, text: str, locale: str = "en-US", rate: float = 1.0) -> TTSResult:
        if not text.strip():
            return LocalWavTTSProvider().synthesize(" ", locale=locale, rate=rate)
        pref = get_runtime_secret("TTS_PROVIDER", settings.tts_provider).lower()
        order = [pref] if pref in self.tts_providers else ["elevenlabs", "google", "local-wav"]
        if "local-wav" not in order:
            order.append("local-wav")

        for key in order:
            prov = self.tts_providers[key]
            if not prov.is_configured():
                continue
            try:
                return prov.synthesize(text=text, locale=locale, rate=rate)
            except Exception as exc:
                logger.warning("TTS provider %s failed, falling back: %s", key, exc)
        return self.tts_providers["local-wav"].synthesize(text=text, locale=locale, rate=rate)
