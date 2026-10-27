"""Tests for the Groq model router.

Uses mocking to test:
- Model catalogue discovery
- Fallback ordering and retry limits
- Error classification (auth, rate limit, model not found, etc.)
- Context-length errors
- The case where every model fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import time

import httpx
import pytest

from src.config import GroqConfig
from src.groq_router import (
    GroqRouter,
    GroqInferenceError,
    GroqContextLengthError,
    ErrorAction,
    ModelCatalog,
    CatalogModel,
    _ApiError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config() -> GroqConfig:
    return GroqConfig(
        api_key="test-key-not-real",
        model_preference=["model-a", "model-b"],
        fallback_models=["model-c"],
        catalog_ttl_seconds=60,
        max_retries=2,
        timeout_seconds=10,
    )


@pytest.fixture
def router(config: GroqConfig) -> GroqRouter:
    r = GroqRouter(config=config)
    r._sleep = lambda _seconds: None
    r._catalog = ModelCatalog(
        models=[CatalogModel(id=model_id) for model_id in ("model-a", "model-b", "model-c", "model-d")],
        fetched_at=time.time(),
    )
    return r


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
class TestErrorClassification:
    def test_401_aborts(self):
        assert GroqRouter.classify_error(401, "Unauthorized") == ErrorAction.ABORT

    def test_403_aborts(self):
        assert GroqRouter.classify_error(403, "Forbidden") == ErrorAction.ABORT

    def test_400_aborts(self):
        assert GroqRouter.classify_error(400, "Bad request") == ErrorAction.ABORT

    def test_400_context_length_reduces(self):
        assert GroqRouter.classify_error(400, "context_length exceeded") == ErrorAction.REDUCE_CONTEXT

    def test_400_too_long_reduces(self):
        assert GroqRouter.classify_error(400, "Input too long for model") == ErrorAction.REDUCE_CONTEXT

    def test_404_tries_next(self):
        assert GroqRouter.classify_error(404, "Model not found") == ErrorAction.TRY_NEXT

    def test_429_retries_same(self):
        assert GroqRouter.classify_error(429, "Rate limited") == ErrorAction.RETRY_SAME

    def test_500_retries_same(self):
        assert GroqRouter.classify_error(500, "Internal server error") == ErrorAction.RETRY_SAME

    def test_503_retries_same(self):
        assert GroqRouter.classify_error(503, "Service unavailable") == ErrorAction.RETRY_SAME


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
class TestModelCatalog:
    def test_empty_catalog_is_expired(self):
        cat = ModelCatalog()
        assert cat.is_expired(60)

    def test_fresh_catalog_not_expired(self):
        cat = ModelCatalog(models=[CatalogModel(id="a")], fetched_at=time.time())
        assert not cat.is_expired(60)

    def test_old_catalog_is_expired(self):
        cat = ModelCatalog(models=[CatalogModel(id="a")], fetched_at=time.time() - 120)
        assert cat.is_expired(60)


class TestCandidateSelection:
    def test_candidates_filtered_by_catalog(self, router: GroqRouter):
        candidates = router.get_available_candidates()
        assert "model-a" in candidates
        assert "model-b" in candidates
        assert "model-c" in candidates
        # model-d is in catalog but not in preferences
        assert "model-d" not in candidates

    def test_preference_order_preserved(self, router: GroqRouter):
        candidates = router.get_available_candidates()
        assert candidates.index("model-a") < candidates.index("model-b")
        assert candidates.index("model-b") < candidates.index("model-c")


# ---------------------------------------------------------------------------
# Inference routing (mocked)
# ---------------------------------------------------------------------------
class TestInferenceRouting:
    def test_success_first_model(self, router: GroqRouter):
        with patch.object(router, "_single_request") as mock:
            from src.groq_router import InferenceResult
            mock.return_value = InferenceResult(
                model_id="model-a", content="Hello", usage={}, latency_ms=100
            )
            result = router.chat_completion([{"role": "user", "content": "test"}])
            assert result.model_id == "model-a"
            assert result.content == "Hello"
            assert result.fallback_used is False

    def test_fallback_on_model_not_found(self, router: GroqRouter):
        from src.groq_router import InferenceResult

        call_count = 0

        def side_effect(model_id, messages, temp, max_tok, resp_fmt):
            nonlocal call_count
            call_count += 1
            if model_id == "model-a":
                raise _ApiError(404, "Model not found", httpx.Headers())
            return InferenceResult(
                model_id=model_id, content="Fallback response", usage={}, latency_ms=50
            )

        with patch.object(router, "_single_request", side_effect=side_effect), patch.object(
            router, "fetch_catalog", return_value=["model-a", "model-b", "model-c", "model-d"]
        ):
            result = router.chat_completion([{"role": "user", "content": "test"}])
            assert result.model_id == "model-b"
            assert result.fallback_used is True

    def test_auth_error_aborts_immediately(self, router: GroqRouter):
        with patch.object(router, "_single_request") as mock:
            mock.side_effect = _ApiError(401, "Invalid API key", httpx.Headers())
            with pytest.raises(GroqInferenceError, match="not sent"):
                router.chat_completion([{"role": "user", "content": "test"}])
            assert mock.call_count == 1

    def test_context_length_error(self, router: GroqRouter):
        with patch.object(router, "_single_request") as mock:
            mock.side_effect = _ApiError(400, "context_length exceeded", httpx.Headers())
            with pytest.raises(GroqContextLengthError):
                router.chat_completion([{"role": "user", "content": "test"}])

    def test_all_models_fail(self, router: GroqRouter):
        with patch.object(router, "_single_request") as mock:
            mock.side_effect = _ApiError(500, "Server error", httpx.Headers())
            with pytest.raises(GroqInferenceError, match="All .* model"):
                router.chat_completion([{"role": "user", "content": "test"}])

    def test_retry_count_bounded(self, router: GroqRouter):
        """Should not exceed max_retries per model."""
        call_count = 0

        def counting_side_effect(model_id, messages, temp, max_tok, resp_fmt):
            nonlocal call_count
            call_count += 1
            raise _ApiError(500, "Server error", httpx.Headers())

        with patch.object(router, "_single_request", side_effect=counting_side_effect):
            with pytest.raises(GroqInferenceError):
                router.chat_completion([{"role": "user", "content": "test"}])

        # 3 candidates × 2 max_retries = 6 total attempts max
        assert call_count <= 6

    def test_malformed_input_error_does_not_cycle(self, router: GroqRouter):
        """HTTP 400 (non-context) should abort, not try next model."""
        with patch.object(router, "_single_request") as mock:
            mock.side_effect = _ApiError(400, "Invalid parameter: xyz", httpx.Headers())
            with pytest.raises(GroqInferenceError, match="was not sent to another model"):
                router.chat_completion([{"role": "user", "content": "test"}])
            assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Pin / unpin
# ---------------------------------------------------------------------------
class TestPinning:
    def test_pin_restricts_candidates(self, router: GroqRouter):
        router.pin_model("model-b")
        from src.groq_router import InferenceResult
        with patch.object(router, "_single_request") as mock:
            mock.return_value = InferenceResult(
                model_id="model-b", content="Pinned", usage={}, latency_ms=50
            )
            result = router.chat_completion([{"role": "user", "content": "test"}])
            assert result.model_id == "model-b"
            # Verify it was called with model-b only
            mock.assert_called_once()
            assert mock.call_args[0][0] == "model-b"

    def test_unpin_restores_routing(self, router: GroqRouter):
        router.pin_model("model-b")
        router.unpin_model()
        candidates = router.get_available_candidates()
        assert len(candidates) > 1


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
class TestStatus:
    def test_status_without_api(self):
        config = GroqConfig(api_key=None)
        router = GroqRouter(config=config)
        status = router.get_status()
        assert status.api_configured is False

    def test_status_with_api(self, router: GroqRouter):
        status = router.get_status()
        assert status.api_configured is True

    def test_no_api_key_raises(self):
        config = GroqConfig(api_key=None)
        router = GroqRouter(config=config)
        with pytest.raises(GroqInferenceError, match="not configured"):
            router.chat_completion([{"role": "user", "content": "test"}])


class TestFallbackPolicy:
    def test_decommissioned_model_tries_next(self, router: GroqRouter):
        from src.groq_router import InferenceResult

        def side_effect(model_id, messages, temp, max_tok, resp_fmt):
            if model_id == "model-a":
                raise _ApiError(
                    400,
                    '{"error":{"code":"model_decommissioned","message":"The model has been decommissioned"}}',
                    httpx.Headers(),
                )
            return InferenceResult(model_id=model_id, content="ok", usage={}, latency_ms=10)

        with patch.object(router, "_single_request", side_effect=side_effect), patch.object(
            router, "fetch_catalog", return_value=["model-b"]
        ) as refresh:
            result = router.chat_completion([{"role": "user", "content": "test"}])
        assert result.model_id == "model-b"
        assert result.fallback_used is True
        refresh.assert_called()

    def test_rate_limit_retries_then_falls_back(self, router: GroqRouter):
        from src.groq_router import InferenceResult

        calls = []

        def side_effect(model_id, messages, temp, max_tok, resp_fmt):
            calls.append(model_id)
            if model_id == "model-a":
                raise _ApiError(
                    429,
                    "Rate limit reached for model `model-a`",
                    httpx.Headers({"retry-after": "1"}),
                )
            return InferenceResult(model_id=model_id, content="later", usage={}, latency_ms=5)

        with patch.object(router, "_single_request", side_effect=side_effect):
            result = router.chat_completion([{"role": "user", "content": "test"}])
        assert result.model_id == "model-b"
        assert calls.count("model-a") == 2  # max_retries on the fixture

    def test_account_wide_limit_does_not_cycle_models(self, router: GroqRouter):
        with patch.object(router, "_single_request") as mock:
            mock.side_effect = _ApiError(
                429,
                "Organization rate limit exceeded for the entire account",
                httpx.Headers({"retry-after": "30"}),
            )
            with pytest.raises(GroqInferenceError, match="account or organization"):
                router.chat_completion([{"role": "user", "content": "test"}])
        assert mock.call_count == 1

    def test_non_chat_models_are_excluded(self, router: GroqRouter):
        router._catalog = ModelCatalog(
            models=[
                CatalogModel(id="model-a"),
                CatalogModel(id="whisper-large-v3"),
                CatalogModel(id="meta-llama/llama-prompt-guard-2-22m", context_window=512),
            ],
            fetched_at=time.time(),
        )
        router.config = GroqConfig(
            api_key="test-key-not-real",
            model_preference=["whisper-large-v3", "meta-llama/llama-prompt-guard-2-22m", "model-a"],
            fallback_models=[],
            catalog_ttl_seconds=60,
            max_retries=1,
            excluded_model_patterns=["whisper", "prompt-guard"],
            min_context_tokens=4096,
        )
        assert router.get_available_candidates() == ["model-a"]

    def test_retry_after_header(self):
        headers = httpx.Headers({"retry-after": "2.5"})
        assert GroqRouter._parse_retry_after(headers) == 2.5

    def test_catalog_parse(self, router: GroqRouter):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                {"id": "llama-3.3-70b-versatile", "active": True, "context_window": 131072, "owned_by": "Meta"},
                {"id": "whisper-large-v3", "active": True, "owned_by": "OpenAI"},
            ]
        }
        client = MagicMock()
        client.get.return_value = response
        with patch.object(router, "_get_client", return_value=client):
            ids = router.fetch_catalog(force=True)
        assert "llama-3.3-70b-versatile" in ids
        assert router._catalog.models[0].context_window == 131072 or any(
            model.context_window == 131072 for model in router._catalog.models
        )
