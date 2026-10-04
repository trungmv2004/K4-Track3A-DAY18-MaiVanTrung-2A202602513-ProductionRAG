"""Provider selection shared by enrichment, generation and RAGAS."""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import urlsplit

import config

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class RequestPacer:
    """Reserve evenly spaced API slots shared by synchronous and async clients."""

    def __init__(self, requests_per_minute, clock=time.monotonic):
        if requests_per_minute <= 0:
            raise ValueError("Requests per minute must be positive")
        self.interval = 60 / requests_per_minute
        self.clock = clock
        self.next_slot = 0.0
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            now = self.clock()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
            return slot - now


_gemini_pacer = RequestPacer(config.GEMINI_REQUESTS_PER_MINUTE)


def _pace_gemini(request):
    if request.url.path.endswith("/chat/completions"):
        time.sleep(_gemini_pacer.reserve())


async def _apace_gemini(request):
    if request.url.path.endswith("/chat/completions"):
        await asyncio.sleep(_gemini_pacer.reserve())


def _gemini_http_clients(timeout):
    import httpx

    return {
        "http_client": httpx.Client(timeout=timeout, event_hooks={"request": [_pace_gemini]}),
        "http_async_client": httpx.AsyncClient(timeout=timeout, event_hooks={"request": [_apace_gemini]}),
    }


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"


def get_settings() -> LLMSettings:
    provider = config.LLM_PROVIDER
    if provider == "auto":
        provider = "gemini" if config.GEMINI_API_KEY and not config.OPENAI_API_KEY else "openai"
    if provider == "gemini":
        return LLMSettings(
            provider,
            config.LLM_MODEL or config.GEMINI_MODEL,
            config.GEMINI_API_KEY,
            GEMINI_BASE_URL,
            config.EVAL_EMBEDDING_MODEL or "gemini-embedding-001",
        )
    if provider == "offline":
        return LLMSettings(provider, "extractive", "", embedding_model=config.EMBEDDING_MODEL)
    if provider == "local":
        url = urlsplit(config.LOCAL_BASE_URL)
        try:
            loopback = url.hostname == "localhost" or ip_address(url.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if (
            not loopback or url.scheme not in {"http", "https"}
            or url.username or url.password or url.query or url.fragment
        ):
            raise ValueError("LOCAL_BASE_URL must point to a loopback HTTP(S) endpoint")
        return LLMSettings(
            provider, config.LLM_MODEL or config.LOCAL_MODEL, "local", config.LOCAL_BASE_URL,
            config.EVAL_EMBEDDING_MODEL or config.EMBEDDING_MODEL,
        )
    if provider == "openai":
        return LLMSettings(
            provider,
            config.LLM_MODEL or "gpt-4o-mini",
            config.OPENAI_API_KEY,
            embedding_model=config.EVAL_EMBEDDING_MODEL or "text-embedding-3-small",
        )
    raise ValueError("LLM_PROVIDER must be auto, openai, gemini, local or offline")


def create_client(timeout: float = 45, max_retries: int = 1):
    from openai import OpenAI

    settings = get_settings()
    kwargs = {"base_url": settings.base_url} if settings.base_url else {}
    if settings.provider == "gemini":
        import httpx

        kwargs["http_client"] = httpx.Client(timeout=timeout, event_hooks={"request": [_pace_gemini]})
    return OpenAI(api_key=settings.api_key, timeout=timeout, max_retries=max_retries, **kwargs)


def safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for key in (config.OPENAI_API_KEY, config.GEMINI_API_KEY):
        if key:
            message = message.replace(key, "[REDACTED]")
    return message


def is_daily_quota_error(message: str) -> bool:
    """Distinguish exhausted daily quota from a recoverable per-minute limit."""
    return "GenerateRequestsPerDay" in message or "requests_per_day" in message.lower()


def check_evaluation_provider():
    """Check chat and evaluation embeddings before spending quota on a full run."""
    settings = get_settings()
    if not settings.api_key:
        raise RuntimeError(f"RAGAS provider {settings.provider} has no generative LLM/API key")
    with create_client(timeout=30, max_retries=0) as client:
        client.chat.completions.create(
            model=settings.model, max_tokens=20,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
    _, embeddings = create_evaluation_models()
    embeddings.embed_query("RAGAS connectivity test")


def create_evaluation_models():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    kwargs = {"base_url": settings.base_url} if settings.base_url else {}
    if settings.provider == "gemini":
        kwargs.update(_gemini_http_clients(120))
    chat = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        temperature=0,
        max_tokens=4096,
        max_retries=1,
        **kwargs,
    )
    if settings.provider == "local":
        from langchain_core.embeddings import Embeddings

        from src.m2_search import _load_encoder

        class LocalEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return _load_encoder(settings.embedding_model).encode(texts).tolist()

            def embed_query(self, text):
                return self.embed_documents([text])[0]

        embeddings = LocalEmbeddings()
    else:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.api_key,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            max_retries=1,
            **kwargs,
        )
    if settings.provider not in {"gemini", "local"}:
        return chat, embeddings

    class SingleCompletionWrapper(LangchainLLMWrapper):
        """Send n separate calls for backends without multiple-completion support."""

        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                temperature=self.get_temperature(n) if temperature is None else temperature,
                stop=stop,
                callbacks=callbacks,
            )
            result.generations = [[g[0] for g in result.generations]]
            return result

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                temperature=self.get_temperature(n) if temperature is None else temperature,
                stop=stop,
                callbacks=callbacks,
            )
            result.generations = [[g[0] for g in result.generations]]
            return result

    return SingleCompletionWrapper(chat), embeddings
