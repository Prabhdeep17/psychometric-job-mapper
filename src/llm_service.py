"""Optional Groq explanations. Dataset scores are never passed back into storage.

If the API key is missing or every model fails, callers receive a structured
error and the deterministic ranking result remains usable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.groq_router import (
    GroqContextLengthError,
    GroqInferenceError,
    GroqRouter,
    InferenceResult,
)
from src.output_validation import (
    invented_score_warnings,
    separate_known_domain_items,
    validate_domain_references,
    validate_jd_analysis_response,
)
from src.prompts import build_explain_prompt, build_jd_analysis_prompt
from src.ranking_engine import RankingResult

logger = logging.getLogger(__name__)

SUGGESTION_DISCLAIMER = (
    "Model-generated suggestion. This is not a stored reference score, "
    "not a psychometric measurement, and not an individual suitability result."
)


@dataclass
class ExplanationResult:
    """Outcome of an optional explanation call."""

    ok: bool
    text: str = ""
    model_id: Optional[str] = None
    fallback_used: bool = False
    latency_ms: float = 0.0
    attempts: int = 0
    context_reduced: bool = False
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class JobDescriptionResult:
    """Job-description suggestions kept separate from stored scores."""

    ok: bool
    model_id: Optional[str] = None
    fallback_used: bool = False
    latency_ms: float = 0.0
    attempts: int = 0
    context_reduced: bool = False
    primary_domains: List[Dict[str, Any]] = field(default_factory=list)
    secondary_domains: List[Dict[str, Any]] = field(default_factory=list)
    unverified_suggestions: List[str] = field(default_factory=list)
    ambiguity_notes: str = ""
    suggested_reference_roles: List[str] = field(default_factory=list)
    review_notes: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    raw_text: str = ""
    disclaimer: str = SUGGESTION_DISCLAIMER


class LlmService:
    """Thin wrapper around the router, prompts, and output checks."""

    def __init__(self, router: GroqRouter) -> None:
        self.router = router

    def explain_ranking(
        self,
        result: RankingResult,
        known_domain_ids: set[str],
        user_question: Optional[str] = None,
    ) -> ExplanationResult:
        if not self.router.config.is_configured:
            return ExplanationResult(
                ok=False,
                error=(
                    "Groq is not configured. The ranking above uses the stored "
                    "dataset only."
                ),
            )
        messages = build_explain_prompt(
            result.role,
            result.role_scope,
            result.domains[:12],
            user_question=user_question,
        )
        logger.info(
            "Explanation requested for role=%s domains=%d",
            result.role,
            min(len(result.domains), 12),
        )
        try:
            inference = self.router.chat_completion(messages, temperature=0.2, max_tokens=1400)
        except GroqContextLengthError as exc:
            return ExplanationResult(ok=False, error=str(exc))
        except GroqInferenceError as exc:
            return ExplanationResult(ok=False, error=str(exc))
        return self._explanation_from_inference(inference, known_domain_ids)

    def analyze_job_description(
        self,
        job_description: str,
        domains: List[Dict[str, str]],
        roles: List[str],
        known_domain_ids: set[str],
        reference_role: Optional[str] = None,
        stored_highlights: Optional[str] = None,
    ) -> JobDescriptionResult:
        text = job_description.strip()
        if not text:
            return JobDescriptionResult(ok=False, error="Paste a job description first.")
        if not self.router.config.is_configured:
            return JobDescriptionResult(
                ok=False,
                error=(
                    "Job-description analysis needs GROQ_API_KEY. "
                    "Stored role mappings still work without it."
                ),
            )
        if len(text) > 7000:
            text = text[:7000] + "\n\n[Truncated by the application before the API call.]"

        logger.info("Job-description analysis requested, characters=%d", len(text))
        messages = build_jd_analysis_prompt(
            text,
            domains,
            roles,
            reference_role=reference_role,
            stored_highlights=stored_highlights,
        )
        try:
            inference = self.router.chat_completion(messages, temperature=0.2, max_tokens=1800)
        except GroqContextLengthError as exc:
            return JobDescriptionResult(ok=False, error=str(exc))
        except GroqInferenceError as exc:
            return JobDescriptionResult(ok=False, error=str(exc))
        return self._jd_from_inference(inference, known_domain_ids)

    def _explanation_from_inference(
        self,
        inference: InferenceResult,
        known_domain_ids: set[str],
    ) -> ExplanationResult:
        validation = validate_domain_references(inference.content, known_domain_ids)
        warnings = list(validation.warnings)
        warnings.extend(invented_score_warnings(inference.content))
        if inference.fallback_used:
            warnings.append(
                f"A fallback model answered ({inference.model_id}). "
                "Quality can differ from the preferred model. "
                "Switching models does not bypass an account-wide Groq quota."
            )
        if inference.context_reduced:
            warnings.append("The application shortened the prompt after a context-length error.")
        return ExplanationResult(
            ok=True,
            text=inference.content,
            model_id=inference.model_id,
            fallback_used=inference.fallback_used,
            latency_ms=inference.latency_ms,
            attempts=inference.attempts,
            context_reduced=inference.context_reduced,
            warnings=warnings,
        )

    def _jd_from_inference(
        self,
        inference: InferenceResult,
        known_domain_ids: set[str],
    ) -> JobDescriptionResult:
        validation = validate_jd_analysis_response(inference.content, known_domain_ids)
        warnings = list(validation.warnings)
        warnings.extend(validation.errors)
        warnings.extend(invented_score_warnings(inference.content))
        if inference.fallback_used:
            warnings.append(
                f"A fallback model answered ({inference.model_id}). "
                "These suggestions are still model-generated. "
                "Switching models does not bypass an account-wide quota."
            )
        data = validation.parsed_data or {}
        primary, unknown_primary = separate_known_domain_items(
            data.get("primary_domains") or [], known_domain_ids
        )
        secondary, unknown_secondary = separate_known_domain_items(
            data.get("secondary_domains") or [], known_domain_ids
        )
        unverified = [str(item) for item in (data.get("unverified_suggestions") or [])]
        unverified.extend(unknown_primary)
        unverified.extend(unknown_secondary)
        roles = data.get("suggested_reference_roles") or []
        if isinstance(roles, str):
            roles = [roles]
        return JobDescriptionResult(
            ok=validation.is_valid and validation.parsed_data is not None,
            model_id=inference.model_id,
            fallback_used=inference.fallback_used,
            latency_ms=inference.latency_ms,
            attempts=inference.attempts,
            context_reduced=inference.context_reduced,
            primary_domains=primary,
            secondary_domains=secondary,
            unverified_suggestions=unverified,
            ambiguity_notes=str(data.get("ambiguity_notes") or ""),
            suggested_reference_roles=[str(role) for role in roles],
            review_notes=str(data.get("review_notes") or ""),
            warnings=warnings,
            error=None if validation.parsed_data else "The model did not return readable JSON.",
            raw_text=inference.content,
        )
