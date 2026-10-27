"""LLM output validation.

Validates that LLM-generated responses:
* Reference only known domain IDs.
* Don't contain invented scores that could be confused with stored ones.
* Parse as expected JSON when structured output is required.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""

    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    parsed_data: Optional[Dict[str, Any]] = None


def validate_domain_references(
    text: str,
    known_domain_ids: Set[str],
) -> ValidationResult:
    """
    Check that any domain IDs mentioned in the text are from the known set.
    """
    # Find all D## patterns
    mentioned = set(re.findall(r"\bD\d{2}\b", text))
    unknown = mentioned - known_domain_ids
    warnings = []
    if unknown:
        warnings.append(
            f"LLM referenced unknown domain IDs: {sorted(unknown)}. "
            f"These are not in the reference dataset."
        )
    return ValidationResult(
        is_valid=len(unknown) == 0,
        warnings=warnings,
    )


def validate_jd_analysis_response(
    raw_text: str,
    known_domain_ids: Set[str],
) -> ValidationResult:
    """
    Validate and parse a JD-analysis JSON response.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Extract JSON from the response (may be wrapped in markdown code blocks)
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        # Remove markdown code fences
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        # Try to find JSON within the text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Could not parse JSON from LLM response: {e}"],
                )
        else:
            return ValidationResult(
                is_valid=False,
                errors=[f"No JSON found in LLM response: {e}"],
            )

    # Validate structure
    expected_keys = {"primary_domains", "secondary_domains"}
    missing_keys = expected_keys - set(data.keys())
    if missing_keys:
        warnings.append(f"Missing expected keys: {missing_keys}")

    # Check domain IDs
    for section in ("primary_domains", "secondary_domains"):
        for item in data.get(section, []):
            did = item.get("domain_id", "")
            if did and did not in known_domain_ids:
                warnings.append(
                    f"LLM suggested unknown domain '{did}' in {section}. "
                    f"This is not in the reference dataset."
                )

    return ValidationResult(
        is_valid=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        parsed_data=data,
    )


def invented_score_warnings(text: str) -> List[str]:
    """Flag phrasing that looks like a model-assigned 1–5 score.

    The text is not rewritten. Stored dataset scores stay authoritative.
    """
    patterns = (
        r"I (?:would |will )?(?:rate|score|assign).*?\d\s*/\s*5",
        r"(?:my|the model's?) score:?\s*\d\s*/\s*5",
        r"relevance score should be\s*\d",
    )
    warnings: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(
                "The model text appears to assign its own 1–5 score. "
                "Only the stored dataset score is the reference value."
            )
            break
    return warnings


def sanitize_llm_output(text: str) -> str:
    """Return model text unchanged.

    Score-invention checks live in ``invented_score_warnings`` so the UI
    can show them without silently editing the model response.
    """
    return text


def separate_known_domain_items(
    items: List[Dict[str, Any]],
    known_domain_ids: Set[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Split suggested domains into known IDs and unverified suggestions.

    Numeric ``score`` fields are removed so a model cannot present a new
    relevance score next to a stored domain.
    """
    known: List[Dict[str, Any]] = []
    unverified: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            unverified.append(str(item))
            continue
        cleaned = {key: value for key, value in item.items() if key.lower() != "score"}
        domain_id = str(cleaned.get("domain_id", "")).strip()
        if domain_id in known_domain_ids:
            known.append(cleaned)
        else:
            label = cleaned.get("domain_name") or domain_id or "unnamed suggestion"
            unverified.append(
                f"{label}: {cleaned.get('reasoning', 'proposed without a known domain ID')}"
            )
    return known, unverified
