"""Turn a typed chat message into a grounded reply.

The dataset still decides the 1–5 scores. The language model only explains
those scores in conversation. If Groq is unavailable, the reply is the
stored ranking in plain text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.data_loader import DatasetLoader
from src.groq_router import GroqInferenceError, GroqRouter
from src.ranking_engine import RankingResult, RankedDomain, rank_domains
from src.role_matcher import ROLE_ALIASES, MatchType, RoleMatch, match_role, _normalize

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are a chat assistant for a reference dataset of job roles and work-related domains.

Write like a person in a chat. Use short paragraphs and plain language.
Do not open with a heading or a memo.

Rules:
- Use only the 1–5 scores printed in the DATA block. Never invent or change a score.
- Do not invent studies, psychometric tests, or citations.
- Do not diagnose personality or mental health.
- Do not say that a score means a person should be hired or will perform well.
- Say briefly that the 1–5 figures are provisional estimates for a role, not a personal profile.
- If DATA says the title is a proxy, say that in the first sentence.
- If DATA says no role was matched, ask the user to choose one of the listed roles. Do not pretend a role was verified.
- If a job description is included, call comments about it model-generated suggestions. Keep them separate from the stored scores.
- Refer to domain names from DATA. Do not introduce a new domain ID.
"""

_JD_MARKERS = (
    "responsibilit",
    "requirement",
    "qualification",
    "we are looking",
    "you will",
    "job description",
    "about the role",
)
_FOLLOW_UP = re.compile(
    r"\b(why|how|explain|more|that|those|this|score|domain|which|what about|tell me|compare|matter)\b",
    re.IGNORECASE,
)


@dataclass
class ChatContext:
    """Facts retrieved before any model call."""

    user_text: str
    matched_role: Optional[str]
    match_type: str
    proxy_warning: Optional[str]
    alternatives: List[str] = field(default_factory=list)
    is_job_description: bool = False
    data_block: str = ""
    offline_reply: str = ""
    known_domain_ids: List[str] = field(default_factory=list)


def looks_like_job_description(text: str) -> bool:
    """A pasted posting is much longer than a title or a short question."""
    lowered = text.lower()
    if len(text) >= 400:
        return True
    if text.count("\n") >= 3 and len(text) >= 180:
        return True
    return any(marker in lowered for marker in _JD_MARKERS)


def _roles_named_in_text(text: str, roles: Sequence[str]) -> List[str]:
    """Return stored role names that appear in the message, longest first."""
    remaining = f" {_normalize(text)} "
    found: List[str] = []
    for role in sorted(roles, key=len, reverse=True):
        token = _normalize(role)
        if token and token in remaining:
            found.append(role)
            remaining = remaining.replace(token, " ")
    return found


def _alias_in_text(text: str, roles: Sequence[str]) -> Optional[RoleMatch]:
    remaining = _normalize(text)
    known = {_normalize(role): role for role in roles}
    for alias in sorted(ROLE_ALIASES, key=len, reverse=True):
        if alias in remaining:
            target = ROLE_ALIASES[alias]
            if _normalize(target) in known:
                return RoleMatch(
                    input_title=text,
                    matched_role=known[_normalize(target)],
                    match_type=MatchType.ALIAS,
                    confidence=0.85,
                    alternatives=[],
                    proxy_warning=(
                        f"'{alias}' is not a stored reference role. "
                        f"Using '{known[_normalize(target)]}' as a proxy. "
                        "The scores belong to that stored role."
                    ),
                )
    return None


def _looks_like_title(text: str) -> bool:
    """A title is a few words, not a question."""
    words = text.strip().split()
    if not words or len(words) > 6:
        return False
    if "?" in text:
        return False
    return True


def resolve_message(
    text: str,
    roles: Sequence[str],
    previous_role: Optional[str],
) -> RoleMatch:
    """Pick a reference role from a chat message, or keep the previous one."""
    named = _roles_named_in_text(text, roles)
    if named:
        return RoleMatch(
            input_title=text,
            matched_role=named[0],
            match_type=MatchType.EXACT,
            confidence=1.0,
            alternatives=named[1:4],
        )

    alias = _alias_in_text(text, roles)
    if alias is not None and not looks_like_job_description(text):
        return alias

    if not looks_like_job_description(text) and _looks_like_title(text):
        return match_role(text, list(roles))

    if previous_role and (_FOLLOW_UP.search(text) or "?" in text):
        return RoleMatch(
            input_title=text,
            matched_role=previous_role,
            match_type=MatchType.EXACT,
            confidence=1.0,
            alternatives=[],
        )

    if looks_like_job_description(text):
        return RoleMatch(
            input_title=text,
            matched_role=None,
            match_type=MatchType.NONE,
            confidence=0.0,
            alternatives=list(roles[:8]),
            proxy_warning="No reference role was named in this description.",
        )

    return match_role(text, list(roles))


def _format_ranking(result: RankingResult, limit: int = 12) -> str:
    lines = []
    for domain in result.domains[:limit]:
        lines.append(
            f"- {domain.domain_id} {domain.domain_name}: stored score "
            f"{domain.relevance_score}/5 ({domain.score_interpretation}). "
            f"{domain.rationale}"
        )
    extra = len(result.domains) - limit
    if extra > 0:
        lines.append(f"- {extra} more domains are in the stored list and were omitted for length.")
    return "\n".join(lines)


def build_context(
    text: str,
    loader: DatasetLoader,
    previous_role: Optional[str] = None,
) -> ChatContext:
    """Retrieve stored rows for this message. Does not call Groq."""
    roles = loader.get_roles()
    match = resolve_message(text, roles, previous_role)
    is_jd = looks_like_job_description(text)
    ranking = None
    if match.matched_role:
        ranking = rank_domains(match.matched_role, loader, top_n=50)

    parts = [f"User message is a job description: {'yes' if is_jd else 'no'}."]
    if match.matched_role:
        label = {
            MatchType.EXACT: "exact stored role",
            MatchType.ALIAS: "alias proxy",
            MatchType.FUZZY: "fuzzy proxy",
            MatchType.NONE: "no match",
        }[match.match_type]
        parts.append(f"Matched reference role: {match.matched_role} ({label}).")
        if match.proxy_warning:
            parts.append(f"Proxy warning: {match.proxy_warning}")
        if ranking is not None:
            parts.append(f"Role scope: {ranking.role_scope}")
            parts.append(
                "Stored domains, highest score first. Ties share a rank. "
                "These numbers are fixed:"
            )
            parts.append(_format_ranking(ranking))
    else:
        parts.append("Matched reference role: none.")
        if match.alternatives:
            parts.append("Closest stored roles: " + ", ".join(match.alternatives) + ".")
        if match.proxy_warning:
            parts.append(match.proxy_warning)

    if is_jd:
        excerpt = text.strip()
        if len(excerpt) > 3500:
            excerpt = excerpt[:3500] + "\n[truncated]"
        parts.append("Job description text:\n" + excerpt)

    data_block = "\n".join(parts)
    offline = _offline_reply(match, ranking, text)
    known_ids = [item["domain_id"] for item in loader.get_domains()]
    return ChatContext(
        user_text=text.strip(),
        matched_role=match.matched_role,
        match_type=match.match_type.value,
        proxy_warning=match.proxy_warning,
        alternatives=list(match.alternatives),
        is_job_description=is_jd,
        data_block=data_block,
        offline_reply=offline,
        known_domain_ids=known_ids,
    )


def _domains_named_in_text(text: str, ranking: RankingResult) -> List:
    """Domains the user named. Generic words like 'domains' do not count."""
    remaining = f" {_normalize(text)} "
    found = []
    ordered = sorted(ranking.domains, key=lambda item: len(item.domain_name), reverse=True)
    for domain in ordered:
        token = _normalize(domain.domain_name)
        if token and token in remaining:
            found.append(domain)
            remaining = remaining.replace(token, " ")
        elif domain.domain_id.lower() in remaining:
            found.append(domain)
    return found


def _domains_to_show(text: str, ranking: RankingResult) -> List:
    named = _domains_named_in_text(text, ranking)
    if named and not re.search(r"\b(most relevant|which domains|what domains|top domains)\b", text, re.I):
        return named
    top_score = ranking.domains[0].relevance_score
    if re.search(r"\b(show more|more domains|next domains|the rest|all domains)\b", text, re.I):
        return [item for item in ranking.domains if item.relevance_score >= max(1, top_score - 1)]
    chosen = [item for item in ranking.domains if item.relevance_score == top_score]
    if len(chosen) < 5:
        chosen = [item for item in ranking.domains if item.relevance_score >= max(1, top_score - 1)]
    return chosen


_BANNED_WHY = re.compile(
    r"it is core to the .+ role|it strongly supports the .+ role|"
    r"directly supports reliable performance|in the role scope",
    re.IGNORECASE,
)

WHY_SYSTEM_PROMPT = """You explain why each stored work domain fits one stored job.

Use only the role scope and that domain's definition. Write 1 or 2 sentences.
Say what this role actually does, using a duty from the role scope, and how this domain's definition carries out that duty.
Do not only repeat the definition. Mention the domain by name. Make each reason different.

Do not invent tasks, studies, tests, citations, or scores.
Do not write "It is core to the ROLE role."
Do not write "directly supports reliable performance" or "in the role scope."
Do not give hiring or personality advice.

Return JSON only:
{"reasons": [{"domain_id": "D01", "why": "one or two sentences"}]}
"""


def _parse_reasons(raw: str) -> Dict[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    reasons = data.get("reasons", []) if isinstance(data, dict) else []
    parsed: Dict[str, str] = {}
    if not isinstance(reasons, list):
        return parsed
    for item in reasons:
        if not isinstance(item, dict):
            continue
        domain_id = str(item.get("domain_id", "")).strip()
        why = str(item.get("why", "")).strip()
        if domain_id and why and not _BANNED_WHY.search(why):
            parsed[domain_id] = why
    return parsed


def _fallback_why(role: str, role_scope: str, domain: RankedDomain) -> str:
    """A specific reason from the stored role scope and this domain's definition."""
    definition = (domain.operational_definition or "").strip().rstrip(".")
    scope = (role_scope or "").strip().rstrip(".")
    if scope and scope[0].isupper():
        scope = scope[0].lower() + scope[1:]
    if definition and definition[0].isupper():
        definition = definition[0].lower() + definition[1:]
    if scope and definition:
        return (
            f"{domain.domain_name} fits {role} because that role {scope}. "
            f"Doing that work depends on {definition}."
        )
    if definition:
        return f"{domain.domain_name} fits {role} because it is {definition}."
    return f"{domain.domain_name} fits the stored work of {role}."


def _explain_why(
    router: GroqRouter,
    role: str,
    role_scope: str,
    domains: Sequence[RankedDomain],
) -> tuple[Dict[str, str], List[str]]:
    """Ask Groq for a specific reason per domain. Scores stay in the dataset."""
    if not domains:
        return {}, []
    if not router.config.is_configured:
        return {}, ["A Groq key is needed to write why each domain fits. The score and definition are from the dataset."]
    lines = [f"Role: {role}", f"Role scope: {role_scope}", "Domains:"]
    for domain in domains:
        lines.append(
            f"- {domain.domain_id} {domain.domain_name}: {domain.operational_definition}"
        )
    try:
        result = router.chat_completion(
            [
                {"role": "system", "content": WHY_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.info("Why-it-fits call failed: %s", type(exc).__name__)
        fallback = {
            domain.domain_id: _fallback_why(role, role_scope, domain)
            for domain in domains
        }
        return fallback, []
    reasons = _parse_reasons(result.content or "")
    missing = [domain.domain_name for domain in domains if domain.domain_id not in reasons]
    warnings = []
    if missing:
        warnings.append("No specific reason was returned for: " + ", ".join(missing) + ".")
    return reasons, warnings


def _offline_reply(
    match: RoleMatch,
    ranking: Optional[RankingResult],
    text: str,
    reasons: Optional[Dict[str, str]] = None,
) -> str:
    if ranking is None or not match.matched_role:
        names = ", ".join(match.alternatives) if match.alternatives else "the role list"
        return (
            "I could not match that to a stored reference role, so I will not invent domains. "
            f"Closest stored roles: {names}. "
            "Type one of those role names."
        )
    shown = _domains_to_show(text, ranking)
    lines = [f"**{match.matched_role}**"]
    if match.match_type == MatchType.ALIAS:
        lines.append(f"Using this role for “{match.input_title}”.")
    elif match.match_type == MatchType.FUZZY:
        lines.append("Closest stored role for the title you typed.")
    reasons = reasons or {}
    for domain in shown:
        block = (
            f"**{domain.domain_name}**\n\n"
            f"Score: {domain.relevance_score}/5, {domain.score_interpretation}\n\n"
            f"What it means: {domain.operational_definition}"
        )
        why = reasons.get(domain.domain_id, "").strip()
        if why:
            block += f"\n\nWhy it fits: {why}"
        lines.append(block)
    return "\n\n".join(lines)


def build_chat_messages(
    context: ChatContext,
    history: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Messages for one Groq call. History is prior user and assistant text only."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for item in list(history)[-6:]:
        role = item.get("role", "user")
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content", "")
        if len(content) > 1200:
            content = content[:1200] + "…"
        messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": (
                "DATA\n"
                f"{context.data_block}\n\n"
                "USER QUESTION\n"
                f"{context.user_text}"
            ),
        }
    )
    return messages


def reply_to_chat(
    text: str,
    loader: DatasetLoader,
    router: GroqRouter,
    history: Sequence[Dict[str, str]],
    previous_role: Optional[str] = None,
) -> tuple[str, ChatContext, List[str]]:
    """Return assistant text, the retrieved context, and any warnings.

    Logs the message length, not the message text.
    """
    logger.info("Chat turn, characters=%d", len(text))
    del history
    context = build_context(text, loader, previous_role=previous_role)
    if context.matched_role is None:
        return context.offline_reply, context, []
    ranking = rank_domains(context.matched_role, loader, top_n=50)
    if ranking is None:
        return context.offline_reply, context, []
    match = resolve_message(text, loader.get_roles(), previous_role)
    shown = _domains_to_show(text, ranking)
    reasons, warnings = _explain_why(router, ranking.role, ranking.role_scope, shown)
    answer = _offline_reply(match, ranking, text, reasons)
    return answer, context, warnings
