"""Groq model registry and automatic fallback router.

The router talks to Groq's OpenAI-compatible endpoint:

* ``GET  {base}/models``
* ``POST {base}/chat/completions``

Model IDs are not hard-coded as the only source of truth. Configured
preferences are checked against the live catalogue. Audio, speech, and
safety-classifier models are skipped because this application only sends
text chat requests.

Fallback policy
---------------
* Model missing, retired, or decommissioned: refresh the catalogue and try
  the next eligible model.
* HTTP 429 that names a model: retry with Retry-After (capped), then try
  another model. This does not bypass an account-wide quota.
* HTTP 429 that is account- or organization-wide: stop and tell the user
  to wait or change quota. Do not cycle models.
* HTTP 5xx, timeouts, and connection errors: retry, then try a fallback.
* HTTP 400 for malformed input or unsupported parameters: surface the error.
* HTTP 401 or 403: stop. Do not try other models.
* Context-length errors: shorten the user text once, then stop if it still
  does not fit.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.config import GroqConfig, get_config

logger = logging.getLogger(__name__)


class ErrorAction(str, Enum):
    """How the router should react to a specific API error."""

    RETRY_SAME = "retry_same"
    TRY_NEXT = "try_next"
    ABORT = "abort"
    ACCOUNT_LIMIT = "account_limit"
    REDUCE_CONTEXT = "reduce_context"


@dataclass
class CatalogModel:
    """One model row from ``GET /models``."""

    id: str
    active: bool = True
    context_window: Optional[int] = None
    owned_by: str = ""


@dataclass
class InferenceError:
    """Structured error from an inference attempt."""

    status_code: int
    message: str
    action: ErrorAction
    model_id: Optional[str] = None
    retry_after: Optional[float] = None


@dataclass
class InferenceResult:
    """Successful inference result."""

    model_id: str
    content: str
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    fallback_used: bool = False
    attempts: int = 1
    context_reduced: bool = False


@dataclass
class ModelStatus:
    """Runtime status of the model router for the Model Status page."""

    api_configured: bool
    current_model: Optional[str]
    available_models: List[str]
    eligible_models: List[str]
    catalog_age_seconds: Optional[float]
    last_error: Optional[str]
    fallback_occurred: bool
    pinned_model: Optional[str]
    catalog_fetched: bool
    skipped_models: List[str] = field(default_factory=list)


@dataclass
class ModelCatalog:
    """Cached Groq model catalogue."""

    models: List[CatalogModel] = field(default_factory=list)
    fetched_at: float = 0.0

    @property
    def model_ids(self) -> List[str]:
        return [model.id for model in self.models]

    def is_expired(self, ttl: int) -> bool:
        if not self.models:
            return True
        return (time.time() - self.fetched_at) > ttl

    @property
    def age_seconds(self) -> float:
        if self.fetched_at == 0:
            return -1
        return time.time() - self.fetched_at


class GroqRouter:
    """Groq model router with bounded retry and automatic fallback."""

    def __init__(self, config: Optional[GroqConfig] = None) -> None:
        self.config = config or get_config().groq
        self._catalog = ModelCatalog()
        self._current_model: Optional[str] = None
        self._last_error: Optional[str] = None
        self._fallback_occurred: bool = False
        self._pinned_model: Optional[str] = None
        self._client: Optional[httpx.Client] = None
        self._sleep: Callable[[float], None] = time.sleep

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()

    def _redact(self, text: str) -> str:
        """Remove secrets before a message is stored or logged."""
        cleaned = text or ""
        if self.config.api_key:
            cleaned = cleaned.replace(self.config.api_key, "[REDACTED]")
        cleaned = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", cleaned)
        return cleaned

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------
    def fetch_catalog(self, force: bool = False) -> List[str]:
        """Fetch available models. Returns IDs, or the cached list on failure."""
        if not force and not self._catalog.is_expired(self.config.catalog_ttl_seconds):
            return self._catalog.model_ids

        if not self.config.is_configured:
            logger.warning("Groq API key not configured; catalogue unavailable.")
            return []

        try:
            client = self._get_client()
            response = client.get("/models")
            response.raise_for_status()
            payload = response.json()
            models: List[CatalogModel] = []
            for row in payload.get("data", []):
                model_id = str(row.get("id", "")).strip()
                if not model_id:
                    continue
                context = row.get("context_window")
                try:
                    context_window = int(context) if context is not None else None
                except (TypeError, ValueError):
                    context_window = None
                models.append(
                    CatalogModel(
                        id=model_id,
                        active=bool(row.get("active", True)),
                        context_window=context_window,
                        owned_by=str(row.get("owned_by", "")),
                    )
                )
            models.sort(key=lambda item: item.id)
            self._catalog = ModelCatalog(models=models, fetched_at=time.time())
            logger.info("Fetched %d models from Groq catalogue.", len(models))
            return self._catalog.model_ids
        except Exception as exc:
            safe = self._redact(str(exc))
            logger.error("Failed to fetch model catalogue: %s", safe)
            self._last_error = f"Catalogue fetch failed: {safe}"[:500]
            return self._catalog.model_ids

    def _is_text_chat_candidate(self, model: CatalogModel) -> bool:
        if not model.active:
            return False
        lowered = model.id.lower()
        for pattern in self.config.excluded_model_patterns:
            if pattern.lower() in lowered:
                return False
        if (
            model.context_window is not None
            and model.context_window < self.config.min_context_tokens
        ):
            return False
        return True

    def get_available_candidates(self) -> List[str]:
        """Configured models that are present and suitable for text chat.

        If the catalogue cannot be fetched, configured IDs are returned
        after the non-chat filter so offline tests and temporary catalogue
        outages can still attempt a request. The error from that request
        is what the UI shows.
        """
        self.fetch_catalog()
        configured = self.config.all_candidate_models
        if not self._catalog.models:
            return [
                model_id
                for model_id in configured
                if self._is_text_chat_candidate(CatalogModel(id=model_id))
            ]

        by_id = {model.id: model for model in self._catalog.models}
        eligible: List[str] = []
        for model_id in configured:
            model = by_id.get(model_id)
            if model is None:
                continue
            if self._is_text_chat_candidate(model):
                eligible.append(model_id)
        return eligible

    def skipped_configured_models(self) -> List[str]:
        """Configured IDs that are missing, inactive, or not text-chat models."""
        if not self._catalog.models:
            return []
        eligible = set(self.get_available_candidates())
        return [model_id for model_id in self.config.all_candidate_models if model_id not in eligible]

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_missing_model(body: str) -> bool:
        lowered = body.lower()
        markers = (
            "model_decommissioned",
            "model_not_found",
            "decommissioned",
            "no longer supported",
            "does not exist",
            "model not found",
            "unknown model",
            "invalid model",
            "is not supported",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_like_context_limit(body: str) -> bool:
        lowered = body.lower()
        markers = (
            "context_length",
            "context length",
            "too long",
            "maximum context",
            "reduce the length",
            "request too large",
            "payload too large",
            "tokens exceed",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_account_wide(body: str) -> bool:
        """True when a 429 does not identify a single model that can be swapped."""
        lowered = body.lower()
        account_markers = (
            "organization rate",
            "org rate",
            "account-wide",
            "account wide",
            "entire account",
            "project rate limit",
            "limit for the organization",
        )
        names_a_model = (
            "for model" in lowered
            or "model `" in lowered
            or "model '" in lowered
            or 'model "' in lowered
        )
        if any(marker in lowered for marker in account_markers) and not names_a_model:
            return True
        return False

    @classmethod
    def classify_error(cls, status_code: int, body: str) -> ErrorAction:
        """Determine the appropriate action for an API error."""
        if status_code in (401, 403):
            return ErrorAction.ABORT
        if status_code in (400, 404, 422):
            if cls._looks_like_missing_model(body):
                return ErrorAction.TRY_NEXT
            if cls._looks_like_context_limit(body):
                return ErrorAction.REDUCE_CONTEXT
            return ErrorAction.ABORT
        if status_code == 413:
            return ErrorAction.REDUCE_CONTEXT
        if status_code == 429:
            if cls._looks_account_wide(body):
                return ErrorAction.ACCOUNT_LIMIT
            return ErrorAction.RETRY_SAME
        if status_code in (408, 409) or status_code >= 500:
            return ErrorAction.RETRY_SAME
        return ErrorAction.ABORT

    @staticmethod
    def _parse_retry_after(headers: httpx.Headers) -> Optional[float]:
        """Read Retry-After or a Groq rate-limit reset header, in seconds."""
        for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            raw = headers.get(key)
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            # Some reset headers are milliseconds.
            if key.startswith("x-ratelimit") and value > 1000:
                value = value / 1000.0
            return value
        return None

    def pin_model(self, model_id: str) -> None:
        self._pinned_model = model_id

    def unpin_model(self) -> None:
        self._pinned_model = None

    def _shrink_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Shorten the longest user message once. System instructions stay intact."""
        shrunk: List[Dict[str, str]] = []
        longest_index = None
        longest_len = 0
        for index, message in enumerate(messages):
            content = message.get("content", "")
            if message.get("role") == "user" and len(content) > longest_len:
                longest_len = len(content)
                longest_index = index
        for index, message in enumerate(messages):
            content = message.get("content", "")
            if index == longest_index and len(content) > 1500:
                keep = max(1500, int(len(content) * 0.55))
                content = (
                    content[:keep]
                    + "\n\n[The application truncated this text because the model "
                    "rejected it as too long. Ask the user for a shorter description "
                    "if important duties were cut off.]"
                )
            shrunk.append({"role": message.get("role", "user"), "content": content})
        return shrunk

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None,
    ) -> InferenceResult:
        """Run a chat completion, retrying and falling back within configured limits.

        Raises GroqInferenceError, GroqContextLengthError, or GroqAuthError.
        """
        if not self.config.is_configured:
            raise GroqInferenceError(
                "Groq API key is not configured. Add GROQ_API_KEY to the .env file. "
                "Role lookup and ranking still work without it."
            )

        if self._pinned_model:
            candidates = [self._pinned_model]
        else:
            candidates = self.get_available_candidates()
            limit = max(1, self.config.max_fallbacks + 1)
            candidates = candidates[:limit]

        if not candidates:
            raise GroqInferenceError(
                "No usable text model is configured. Open the Model Status page, "
                "copy a chat model ID from the live catalogue, and set "
                "GROQ_MODEL_PREFERENCE in .env. Audio, speech, and guard models "
                "are skipped on purpose. Restart the app after changing .env."
            )

        errors: List[InferenceError] = []
        total_attempts = 0
        context_reduced = False
        working_messages = messages

        for model_idx, model_id in enumerate(candidates):
            retries_left = self.config.max_retries

            while retries_left > 0:
                total_attempts += 1
                retries_left -= 1
                try:
                    result = self._single_request(
                        model_id,
                        working_messages,
                        temperature,
                        max_tokens,
                        response_format,
                    )
                    result.fallback_used = model_idx > 0
                    result.attempts = total_attempts
                    result.context_reduced = context_reduced
                    self._current_model = model_id
                    if model_idx > 0:
                        self._fallback_occurred = True
                    self._last_error = None
                    return result

                except _ApiError as api_err:
                    action = self.classify_error(api_err.status_code, api_err.body)
                    retry_after = self._parse_retry_after(api_err.headers)
                    safe_body = self._redact(api_err.body)[:400]
                    errors.append(
                        InferenceError(
                            status_code=api_err.status_code,
                            message=safe_body,
                            action=action,
                            model_id=model_id,
                            retry_after=retry_after,
                        )
                    )
                    self._last_error = f"[{model_id}] HTTP {api_err.status_code}: {safe_body[:240]}"
                    logger.warning(
                        "Inference error on %s: HTTP %s action=%s",
                        model_id,
                        api_err.status_code,
                        action.value,
                    )

                    if action == ErrorAction.ABORT:
                        if api_err.status_code in (401, 403):
                            raise GroqAuthError(
                                "Groq rejected the API key or the account permission "
                                f"(HTTP {api_err.status_code}). The request was not sent "
                                "to another model. Check GROQ_API_KEY and restart the app."
                            )
                        raise GroqInferenceError(
                            f"The request was rejected (HTTP {api_err.status_code}) "
                            f"and was not sent to another model: {safe_body[:300]}"
                        )
                    if action == ErrorAction.ACCOUNT_LIMIT:
                        wait_note = (
                            f" The server asked to wait about {retry_after:.0f} seconds."
                            if retry_after
                            else ""
                        )
                        raise GroqInferenceError(
                            "Groq reported a rate limit that appears to apply to the "
                            "account or organization, not only to one model. Switching "
                            f"models will not bypass that limit.{wait_note} Wait and "
                            "use Retry, or change the account quota."
                        )
                    if action == ErrorAction.REDUCE_CONTEXT:
                        if not context_reduced:
                            working_messages = self._shrink_messages(working_messages)
                            context_reduced = True
                            retries_left += 1
                            continue
                        raise GroqContextLengthError(
                            f"The input is too long for {model_id}, even after the "
                            "application shortened it. Paste a shorter job description."
                        )
                    if action == ErrorAction.TRY_NEXT:
                        self.fetch_catalog(force=True)
                        break

                    wait = 0.0
                    if retry_after is not None:
                        wait = min(max(retry_after, 0.0), self.config.max_retry_wait_seconds)
                        if retry_after > self.config.max_retry_wait_seconds:
                            self._last_error = (
                                f"[{model_id}] HTTP 429. Server asked to wait "
                                f"{retry_after:.0f}s. This app waits at most "
                                f"{self.config.max_retry_wait_seconds:.0f}s, then "
                                "tries another configured model if the limit looks "
                                "model-specific. An account-wide quota still requires waiting."
                            )
                    elif retries_left > 0:
                        wait = min(2 ** (self.config.max_retries - retries_left), 8)
                    if wait > 0 and retries_left > 0:
                        self._sleep(wait)
                    elif action == ErrorAction.RETRY_SAME and retries_left == 0:
                        break

                except httpx.TimeoutException:
                    self._record_transport_error(model_id, "Request timed out", errors)
                    if retries_left > 0:
                        self._sleep(1)

                except httpx.ConnectError as exc:
                    self._record_transport_error(
                        model_id, f"Connection error: {self._redact(str(exc))}", errors
                    )
                    if retries_left > 0:
                        self._sleep(1)

        summary = "; ".join(
            f"{item.model_id}→HTTP {item.status_code}" for item in errors[-5:]
        )
        raise GroqInferenceError(
            f"All {len(candidates)} configured model(s) failed after {total_attempts} "
            f"attempt(s). Last errors: {summary or 'none recorded'}. "
            "If this is an account-wide rate limit, switching models will not help. "
            "Wait, then use Retry, or set GROQ_MODEL_PREFERENCE to a model your "
            "account can call."
        )

    def _record_transport_error(
        self,
        model_id: str,
        message: str,
        errors: List[InferenceError],
    ) -> None:
        self._last_error = f"[{model_id}] {message}"[:500]
        errors.append(
            InferenceError(
                status_code=0,
                message=message[:400],
                action=ErrorAction.RETRY_SAME,
                model_id=model_id,
            )
        )

    def _single_request(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
    ) -> InferenceResult:
        """Make one chat-completion request. Does not log message text."""
        client = self._get_client()
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        started = time.perf_counter()
        response = client.post("/chat/completions", json=payload)
        latency = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            raise _ApiError(response.status_code, response.text, response.headers)

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _ApiError(200, f"Unexpected response shape: {exc}", response.headers) from exc
        if not isinstance(content, str):
            content = str(content)

        return InferenceResult(
            model_id=model_id,
            content=content,
            usage=data.get("usage", {}) or {},
            latency_ms=round(latency, 1),
        )

    def get_status(self) -> ModelStatus:
        """Return current router status. Does not call the network by itself."""
        eligible = []
        skipped: List[str] = []
        if self._catalog.models:
            by_id = {model.id: model for model in self._catalog.models}
            for model_id in self.config.all_candidate_models:
                model = by_id.get(model_id)
                if model and self._is_text_chat_candidate(model):
                    eligible.append(model_id)
                else:
                    skipped.append(model_id)
        return ModelStatus(
            api_configured=self.config.is_configured,
            current_model=self._current_model,
            available_models=self._catalog.model_ids,
            eligible_models=eligible,
            catalog_age_seconds=(
                round(self._catalog.age_seconds, 1) if self._catalog.fetched_at > 0 else None
            ),
            last_error=self._last_error,
            fallback_occurred=self._fallback_occurred,
            pinned_model=self._pinned_model,
            catalog_fetched=self._catalog.fetched_at > 0,
            skipped_models=skipped,
        )


class GroqInferenceError(Exception):
    """Raised when inference fails after the allowed attempts."""


class GroqContextLengthError(GroqInferenceError):
    """Raised when the input exceeds the model's context window."""


class GroqAuthError(GroqInferenceError):
    """Raised for HTTP 401 or 403. Kept for callers that want to branch."""


class _ApiError(Exception):
    """Internal wrapper for a non-success HTTP response."""

    def __init__(self, status_code: int, body: str, headers: httpx.Headers) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers
        super().__init__(f"HTTP {status_code}")
