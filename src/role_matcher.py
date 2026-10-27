"""Role matching and alias resolution.

Maps user-supplied job titles to stored reference roles using:
1. Exact match (case-insensitive)
2. Alias / synonym table
3. Fuzzy string matching as a last resort

When no match is found, returns the closest candidates and labels the
result as a proxy — never silently pretends it is a verified role.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class MatchType(str, Enum):
    """How the user's input was resolved to a reference role."""

    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy_proxy"
    NONE = "no_match"


@dataclass
class RoleMatch:
    """Result of a role-matching attempt."""

    input_title: str
    matched_role: Optional[str]
    match_type: MatchType
    confidence: float  # 0.0–1.0
    alternatives: List[str]
    proxy_warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Static alias / synonym table  (extend as needed)
# ---------------------------------------------------------------------------
ROLE_ALIASES: dict[str, str] = {
    # Data roles
    "data engineer": "Data Analyst",
    "bi analyst": "Data Analyst",
    "business intelligence analyst": "Data Analyst",
    "analytics engineer": "Data Analyst",
    "data analytics": "Data Analyst",
    "statistician": "Data Scientist",
    "applied scientist": "Data Scientist",
    "quantitative analyst": "Data Scientist",
    "ml engineer": "Machine Learning Engineer",
    "ai engineer": "Machine Learning Engineer",
    "deep learning engineer": "Machine Learning Engineer",
    "nlp engineer": "Machine Learning Engineer",
    "mlops engineer": "Machine Learning Engineer",
    # Engineering
    "developer": "Software Engineer",
    "software developer": "Software Engineer",
    "backend engineer": "Software Engineer",
    "frontend engineer": "Software Engineer",
    "full stack developer": "Software Engineer",
    "fullstack developer": "Software Engineer",
    "web developer": "Software Engineer",
    "devops engineer": "Software Engineer",
    "sre": "Software Engineer",
    "site reliability engineer": "Software Engineer",
    "platform engineer": "Software Engineer",
    # Product / PM
    "program manager": "Project Manager",
    "technical program manager": "Project Manager",
    "scrum master": "Project Manager",
    "product owner": "Product Manager",
    "product lead": "Product Manager",
    # Cybersecurity
    "security analyst": "Cybersecurity Analyst",
    "security engineer": "Cybersecurity Analyst",
    "information security analyst": "Cybersecurity Analyst",
    "penetration tester": "Cybersecurity Analyst",
    "soc analyst": "Cybersecurity Analyst",
    # Finance
    "accountant": "Financial Analyst",
    "finance analyst": "Financial Analyst",
    "investment analyst": "Financial Analyst",
    "credit analyst": "Financial Analyst",
    # Research
    "scientist": "Research Scientist",
    "research engineer": "Research Scientist",
    "postdoc": "Research Scientist",
    "postdoctoral researcher": "Research Scientist",
    # Marketing / Sales
    "brand manager": "Marketing Manager",
    "digital marketer": "Marketing Manager",
    "growth marketer": "Marketing Manager",
    "marketing analyst": "Marketing Manager",
    "sales manager": "Sales Representative",
    "account executive": "Sales Representative",
    "business development": "Sales Representative",
    "bdm": "Sales Representative",
    "business development representative": "Sales Representative",
    # HR / Ops
    "recruiter": "HR Manager",
    "talent acquisition": "HR Manager",
    "people operations": "HR Manager",
    "hr specialist": "HR Manager",
    "operations analyst": "Operations Manager",
    "supply chain manager": "Operations Manager",
    # Other
    "ux researcher": "UX Designer",
    "ui designer": "UX Designer",
    "ui/ux designer": "UX Designer",
    "product designer": "UX Designer",
    "interaction designer": "UX Designer",
    "attorney": "Lawyer",
    "legal counsel": "Lawyer",
    "paralegal": "Lawyer",
    "nurse": "Healthcare Professional",
    "doctor": "Healthcare Professional",
    "physician": "Healthcare Professional",
    "therapist": "Healthcare Professional",
    "pharmacist": "Healthcare Professional",
    "professor": "Teacher",
    "instructor": "Teacher",
    "educator": "Teacher",
    "trainer": "Teacher",
    "tutor": "Teacher",
    "founder": "Entrepreneur",
    "startup founder": "Entrepreneur",
    "ceo": "Entrepreneur",
    "business owner": "Entrepreneur",
    "customer service": "Customer Support Specialist",
    "customer service representative": "Customer Support Specialist",
    "support agent": "Customer Support Specialist",
    "technical support": "Customer Support Specialist",
    "help desk": "Customer Support Specialist",
    "civil engineer": "Mechanical Engineer",
    "electrical engineer": "Mechanical Engineer",
    "structural engineer": "Mechanical Engineer",
}


def _normalize(title: str) -> str:
    """Lower-case, strip extra whitespace and punctuation."""
    return re.sub(r"\s+", " ", title.strip().lower())


def _similarity(a: str, b: str) -> float:
    """Sequence-matcher ratio between two normalised strings."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def match_role(
    user_title: str,
    known_roles: List[str],
    fuzzy_threshold: float = 0.55,
    max_alternatives: int = 5,
) -> RoleMatch:
    """
    Resolve a user-supplied job title to a stored reference role.

    Priority:
      1. Exact match (case-insensitive)
      2. Alias lookup
      3. Fuzzy match above threshold
      4. No match — return closest alternatives
    """
    norm = _normalize(user_title)
    role_map = {_normalize(r): r for r in known_roles}

    # 1. Exact
    if norm in role_map:
        return RoleMatch(
            input_title=user_title,
            matched_role=role_map[norm],
            match_type=MatchType.EXACT,
            confidence=1.0,
            alternatives=[],
        )

    # 2. Alias
    if norm in ROLE_ALIASES:
        canonical = ROLE_ALIASES[norm]
        if _normalize(canonical) in role_map:
            return RoleMatch(
                input_title=user_title,
                matched_role=canonical,
                match_type=MatchType.ALIAS,
                confidence=0.85,
                alternatives=[],
                proxy_warning=(
                    f"'{user_title}' is not a stored reference role. "
                    f"Using '{canonical}' as a known alias/proxy. "
                    f"Domain relevance scores are from the proxy role and may "
                    f"not perfectly represent '{user_title}'."
                ),
            )

    # 3. Fuzzy
    scored = [(r, _similarity(user_title, r)) for r in known_roles]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_role, best_score = scored[0]
    alternatives = [r for r, _ in scored[:max_alternatives]]

    if best_score >= fuzzy_threshold:
        return RoleMatch(
            input_title=user_title,
            matched_role=best_role,
            match_type=MatchType.FUZZY,
            confidence=round(best_score, 3),
            alternatives=alternatives,
            proxy_warning=(
                f"'{user_title}' is not a stored reference role. "
                f"The closest match is '{best_role}' "
                f"(similarity {best_score:.0%}). "
                f"Scores shown are from the proxy role."
            ),
        )

    # 4. No match
    return RoleMatch(
        input_title=user_title,
        matched_role=None,
        match_type=MatchType.NONE,
        confidence=0.0,
        alternatives=alternatives,
        proxy_warning=(
            f"'{user_title}' could not be matched to any stored reference role. "
            f"Please select one of the closest roles manually, or browse "
            f"the full role list."
        ),
    )
