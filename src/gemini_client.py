from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from google import genai

LOGGER = logging.getLogger(__name__)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
FALLBACK_STATUS_CODES = TRANSIENT_STATUS_CODES | {404}


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: tuple[str, ...] = (),
        max_retries: int = 1,
        retry_base_seconds: float = 1.0,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Add it to .env before asking questions.")
        if max_retries < 0:
            raise ValueError("GEMINI_MAX_RETRIES cannot be negative")
        if retry_base_seconds < 0:
            raise ValueError("GEMINI_RETRY_BASE_SECONDS cannot be negative")

        self.models = tuple(dict.fromkeys((model, *fallback_models)))
        self.primary_model = model
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    @classmethod
    def _can_fallback(cls, exc: Exception) -> bool:
        status_code = cls._status_code(exc)
        if status_code in FALLBACK_STATUS_CODES:
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "quota",
                "resource_exhausted",
                "high demand",
                "not found",
                "unsupported",
                "unavailable",
            )
        )

    @classmethod
    def _is_transient(cls, exc: Exception) -> bool:
        status_code = cls._status_code(exc)
        if status_code in TRANSIENT_STATUS_CODES:
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("quota", "resource_exhausted", "high demand", "unavailable")
        )

    def generate(self, prompt: str) -> GenerationResult:
        failures: list[str] = []
        for model_index, model in enumerate(self.models):
            if model_index:
                LOGGER.warning("Trying configured Gemini fallback model: %s", model)

            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(model=model, contents=prompt)
                    text = getattr(response, "text", None)
                    if not text:
                        raise RuntimeError("Gemini returned no answer text")
                    return GenerationResult(text=text.strip(), model=model)
                except Exception as exc:
                    status = self._status_code(exc)
                    failures.append(f"{model}: {status or type(exc).__name__} {exc}")

                    if not self._can_fallback(exc):
                        raise RuntimeError(f"Gemini request failed using '{model}': {exc}") from exc

                    if attempt < self.max_retries and self._is_transient(exc):
                        delay = self.retry_base_seconds * (2**attempt)
                        LOGGER.warning(
                            "Gemini model %s failed (%s); retrying in %.1fs (%d/%d)",
                            model,
                            status or type(exc).__name__,
                            delay,
                            attempt + 1,
                            self.max_retries,
                        )
                        time.sleep(delay)
                        continue
                    break

        attempted = ", ".join(self.models)
        latest_by_model = failures[-len(self.models) :]
        raise RuntimeError(
            "All configured Gemini models failed. "
            f"Attempted: {attempted}. Last errors: {'; '.join(latest_by_model)}. "
            "Edit GEMINI_MODEL or GEMINI_FALLBACK_MODELS in .env; "
            "MemoryLens never chooses an unlisted model."
        )
