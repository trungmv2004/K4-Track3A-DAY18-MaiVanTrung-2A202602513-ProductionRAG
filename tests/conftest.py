"""Unit tests use mocks and never spend API quota from the user's .env."""

import pytest

import config


@pytest.fixture(autouse=True)
def isolate_api_keys(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "LLM_MODEL", "")
