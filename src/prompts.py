"""Prompt templates for LLM-powered features.

All prompts enforce:
* The LLM must not invent scores, sources, or citations.
* Output must reference only known domain IDs.
* LLM-generated context adjustments are labelled as suggestions.
* The LLM must not diagnose personality or claim predictive validity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.ranking_engine import RankedDomain


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_EXPLAIN = """You are a psychometric domain relevance expert.

Your task is to explain WHY specific psychometric or work-style domains are
relevant (or not) to a given job role, based ONLY on the information provided.

STRICT RULES:
1. Do NOT invent scores. The relevance scores are provided and fixed.
2. Do NOT invent sources, studies, psychometric tests, or academic citations.
3. Do NOT claim that domain scores predict employee performance unless validated evidence is explicitly provided.
4. Do NOT diagnose personality or mental health conditions.
5. Do NOT treat role-level relevance as individual job suitability.
6. All scores are PROVISIONAL ESTIMATES, not scientifically validated measurements.
7. Reference only the domain IDs and names provided.
8. If uncertain, say so explicitly rather than fabricating an explanation.

Your response should be clear, professional, and helpful."""

SYSTEM_PROMPT_JD_ANALYSIS = """You are a psychometric domain mapping assistant.

Your task is to analyze a job description and identify which of the provided
reference domains are most relevant to the duties and requirements described.

STRICT RULES:
1. Do NOT assign numerical scores. The reference scores are fixed and come from the dataset.
2. Do NOT invent sources, studies, or citations.
3. ONLY reference domain IDs from the provided list.
4. Label ALL your suggestions as "model-generated suggestions" that require human review.
5. If the role is ambiguous, explain why and suggest which reference roles may apply.
6. Do NOT claim predictive validity.
7. Keep analysis grounded in the actual text of the job description.
8. If a domain seems relevant but is not in the reference list, note it as an unverified suggestion.

Respond in the following JSON format:
{
  "primary_domains": [
    {"domain_id": "D01", "domain_name": "...", "reasoning": "..."}
  ],
  "secondary_domains": [
    {"domain_id": "D05", "domain_name": "...", "reasoning": "..."}
  ],
  "ambiguity_notes": "...",
  "suggested_reference_roles": ["..."],
  "review_notes": "...",
  "unverified_suggestions": ["..."]
}"""


# ---------------------------------------------------------------------------
# User-prompt builders
# ---------------------------------------------------------------------------
def build_explain_prompt(
    role: str,
    role_scope: str,
    domains: List[RankedDomain],
    user_question: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build messages for explaining domain relevance for a role.
    """
    domain_block = "\n".join(
        f"- {d.domain_id} {d.domain_name} (Score: {d.relevance_score}/5, "
        f"Rank: {d.rank}, Family: {d.construct_family})\n"
        f"  Definition: {d.operational_definition}\n"
        f"  Rationale: {d.rationale}\n"
        f"  Confidence: {d.confidence} | Status: {d.score_status}\n"
        f"  Evidence: {d.evidence_status}\n"
        f"  O*NET ref: {d.onet_reference_occupation} ({d.onet_profile_match_quality})"
        for d in domains
    )

    user_content = (
        f"Role: {role}\n"
        f"Role scope: {role_scope}\n\n"
        f"Top relevant domains (sorted by stored relevance score):\n{domain_block}\n\n"
        f"Please explain why these domains are relevant to this role. "
        f"Organize by importance tier. Note any domains where the evidence is "
        f"weak or the O*NET proxy is only partial."
    )

    if user_question:
        user_content += f"\n\nAdditional question from the user: {user_question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT_EXPLAIN},
        {"role": "user", "content": user_content},
    ]


def build_jd_analysis_prompt(
    job_description: str,
    available_domains: List[Dict[str, str]],
    available_roles: List[str],
    reference_role: Optional[str] = None,
    stored_highlights: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build messages for job-description analysis.

    Stored highlights, when provided, are context only. The prompt tells
    the model not to replace those scores.
    """
    domain_list = "\n".join(
        f"- {d['domain_id']}: {d['domain_name']} [{d['construct_family']}] — "
        f"{d['operational_definition'][:240]}"
        for d in available_domains
    )

    role_list = ", ".join(available_roles)
    role_block = ""
    if reference_role:
        role_block = (
            f"\nThe user selected this reference role: {reference_role}.\n"
            "If the description does not fit that role, say so in ambiguity_notes.\n"
        )
    stored_block = ""
    if stored_highlights:
        stored_block = (
            "\nStored reference scores for that role (fixed; do not change them):\n"
            f"{stored_highlights}\n"
        )

    user_content = (
        f"Job Description:\n{job_description}\n\n"
        f"Available reference domains ({len(available_domains)} total):\n{domain_list}\n\n"
        f"Available reference roles: {role_list}\n"
        f"{role_block}{stored_block}\n"
        "Analyze this job description and identify which reference domains "
        "the duties mention. Suggest the best-matching reference role(s) from "
        "the available list. Label every suggestion as model-generated. "
        "Do not output a 1–5 score. Note ambiguity or missing information. "
        "Put any construct that is not in the domain list under unverified_suggestions."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT_JD_ANALYSIS},
        {"role": "user", "content": user_content},
    ]
