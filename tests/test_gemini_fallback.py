from __future__ import annotations

import unittest

from src.gemini_client import GeminiClient


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class FakeModels:
    def __init__(self, failures: dict[str, Exception]):
        self.failures = failures
        self.calls: list[str] = []

    def generate_content(self, model: str, contents: str):
        self.calls.append(model)
        if model in self.failures:
            raise self.failures[model]
        return type("Response", (), {"text": f"answer from {model}"})()


class FakeClient:
    def __init__(self, models: FakeModels):
        self.models = models


class GeminiFallbackTests(unittest.TestCase):
    def test_falls_back_after_unavailable_primary(self):
        client = GeminiClient(
            "test-key",
            "primary",
            ("fallback",),
            max_retries=0,
        )
        models = FakeModels({"primary": FakeAPIError(503, "high demand")})
        client.client = FakeClient(models)

        result = client.generate("evidence and question")

        self.assertEqual(result.model, "fallback")
        self.assertEqual(result.text, "answer from fallback")
        self.assertEqual(models.calls, ["primary", "fallback"])

    def test_does_not_fallback_on_authentication_error(self):
        client = GeminiClient(
            "test-key",
            "primary",
            ("fallback",),
            max_retries=0,
        )
        models = FakeModels({"primary": FakeAPIError(401, "invalid API key")})
        client.client = FakeClient(models)

        with self.assertRaisesRegex(RuntimeError, "request failed"):
            client.generate("evidence and question")

        self.assertEqual(models.calls, ["primary"])


if __name__ == "__main__":
    unittest.main()
