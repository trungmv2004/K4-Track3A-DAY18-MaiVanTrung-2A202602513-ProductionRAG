import asyncio

import pytest
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import StringPromptValue
from langchain_openai import ChatOpenAI

import config
from src import llm


def test_gemini_auto_selection_and_secret_repr(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "private-test-value")
    settings = llm.get_settings()
    assert settings.provider == "gemini"
    assert settings.model == config.GEMINI_MODEL
    assert settings.base_url == llm.GEMINI_BASE_URL
    assert settings.embedding_model == "gemini-embedding-001"
    assert "private-test-value" not in repr(settings)
    assert "private-test-value" not in llm.safe_error(ValueError("private-test-value"))


def test_explicit_provider_and_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MODEL", "custom-model")
    assert llm.get_settings().model == "custom-model"
    assert llm.get_settings().base_url is None


def test_gemini_multi_generation_contract(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    wrapper, embeddings = llm.create_evaluation_models()
    calls = []

    def generate(self, prompts, **kwargs):
        assert "n" not in kwargs
        calls.append(len(prompts))
        return LLMResult(generations=[[Generation(text=str(i))] for i in range(len(prompts))])

    async def agenerate(self, prompts, **kwargs):
        return generate(self, prompts, **kwargs)

    monkeypatch.setattr(ChatOpenAI, "generate_prompt", generate)
    monkeypatch.setattr(ChatOpenAI, "agenerate_prompt", agenerate)
    prompt = StringPromptValue(text="q")
    sync = wrapper.generate_text(prompt, n=3)
    asynchronous = asyncio.run(wrapper.agenerate_text(prompt, n=3))
    assert [[g.text for g in group] for group in sync.generations] == [["0", "1", "2"]]
    assert len(asynchronous.generations[0]) == 3 and calls == [3, 3]
    assert embeddings.check_embedding_ctx_length is False
    assert embeddings.model_kwargs["encoding_format"] == "float"


def test_offline_provider_ignores_cloud_keys(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "offline")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "configured-but-not-authorized")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "configured-but-not-authorized")
    settings = llm.get_settings()
    assert settings.provider == "offline"
    assert settings.api_key == ""
    assert settings.base_url is None


def test_offline_pipeline_never_constructs_cloud_clients(monkeypatch):
    from src import generation, m4_eval, m5_enrichment

    monkeypatch.setattr(config, "LLM_PROVIDER", "offline")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "configured-but-not-authorized")

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline processing must not construct an API client")

    monkeypatch.setattr(generation, "create_client", forbidden)
    monkeypatch.setattr(m5_enrichment, "create_client", forbidden)
    monkeypatch.setattr(m4_eval, "create_evaluation_models", forbidden)
    assert generation.generate_answer("Q", ["Context"]) == ("Context", "extractive_no_api_key")
    assert m5_enrichment.enrich_chunks([{"text": "Context"}])[0].auto_metadata[
        "enrichment_backend"
    ] == "extractive"
    results = m4_eval.evaluate_ragas(["Q"], ["Context"], [["Context"]], ["Truth"])
    assert results["evaluation_status"] == "unavailable"
    assert "Offline mode" in results["evaluation_error"]


def test_local_provider_uses_loopback_and_local_embeddings(monkeypatch):
    import numpy as np

    from src import m2_search

    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "cloud-secret")
    settings = llm.get_settings()
    assert settings.api_key == "local"
    assert settings.embedding_model == config.EMBEDDING_MODEL

    class Encoder:
        def encode(self, texts):
            return np.array([[len(text), 1.0] for text in texts])

    monkeypatch.setattr(m2_search, "_load_encoder", lambda model: Encoder())
    wrapper, embeddings = llm.create_evaluation_models()
    assert wrapper.langchain_llm.openai_api_base == settings.base_url
    assert embeddings.embed_documents(["abc", "xy"]) == [[3.0, 1.0], [2.0, 1.0]]
    assert embeddings.embed_query("abcd") == [4.0, 1.0]


@pytest.mark.parametrize("url", [
    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "http://192.168.1.2:11434/v1", "http://127.0.0.1@example.com/v1",
])
def test_local_provider_rejects_non_loopback_urls(monkeypatch, url):
    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    monkeypatch.setattr(config, "LOCAL_BASE_URL", url)
    with pytest.raises(ValueError, match="loopback"):
        llm.get_settings()


def test_request_pacer_reserves_shared_slots_without_sleeping():
    now = [100.0]
    pacer = llm.RequestPacer(15, clock=lambda: now[0])
    assert pacer.reserve() == 0
    assert pacer.reserve() == 4
    assert pacer.reserve() == 8
    now[0] = 120
    assert pacer.reserve() == 0
    with pytest.raises(ValueError):
        llm.RequestPacer(0)


def test_sync_and_async_hooks_share_quota_without_throttling_embeddings(monkeypatch):
    import httpx

    delays = []
    monkeypatch.setattr(llm, "_gemini_pacer", llm.RequestPacer(15, clock=lambda: 100))
    monkeypatch.setattr(llm.time, "sleep", delays.append)

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    chat = httpx.Request("POST", llm.GEMINI_BASE_URL + "chat/completions")
    embedding = httpx.Request("POST", llm.GEMINI_BASE_URL + "embeddings")
    llm._pace_gemini(chat)
    asyncio.run(llm._apace_gemini(chat))
    llm._pace_gemini(embedding)
    asyncio.run(llm._apace_gemini(embedding))
    assert delays == [0, 4]


def test_gemini_sdk_retries_also_use_shared_request_pacer(monkeypatch):
    import httpx

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    seen = []
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setattr(llm, "_gemini_pacer", llm.RequestPacer(15, clock=lambda: 100))

    def respond(request):
        seen.append(request.url.path)
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"message": "quota", "type": "rate_limit"}},
                                  headers={"retry-after": "0"})
        return httpx.Response(200, json={
            "id": "test", "created": 0, "model": "test", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
        })

    original_client = httpx.Client

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", MockClient)
    with llm.create_client(max_retries=1) as client:
        response = client.chat.completions.create(model="test", messages=[{"role": "user", "content": "q"}])
    assert response.choices[0].message.content == "ok"
    assert seen == ["/v1beta/openai/chat/completions"] * 2
    assert 4 in waits  # The retry reserved the next quota slot rather than bypassing the pacer.
