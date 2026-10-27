"""Psychometric Domain Mapping — Streamlit application.

Dataset lookup and ranking work without a Groq API key. Explanations and
job-description analysis are optional and never overwrite stored scores.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.chat_service import reply_to_chat
from src.config import get_config
from src.data_loader import DataLoadError, DatasetLoader
from src.groq_router import GroqRouter
from src.llm_service import LlmService
from src.ranking_engine import (
    RankedDomain,
    RankingResult,
    domains_to_dataframe,
    filter_domains,
    get_full_profile,
    rank_domains,
)
from src.review_flags import add_flag, clear_flags, list_flags
from src.role_matcher import MatchType, RoleMatch, match_role
from src.workbook_loader import (
    AUDIT_SHEET,
    BENCHMARKS_SHEET,
    CROSSWALK_SHEET,
    QA_SHEET,
    REVIEW_QUEUE_SHEET,
    SCORING_SHEET,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROVISIONAL_NOTE = (
    "Relevance scores are provisional estimates stored in the reference dataset. "
    "They are not scientifically validated psychometric measurements, not an "
    "individual psychological profile, and not a hiring or suitability score. "
    "O*NET 0–100 work-style benchmarks are a separate source and are not converted "
    "into the 1–5 scores."
)

st.set_page_config(
    page_title="Psychometric Domain Mapping",
    page_icon="PM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
    .block-title { font-size: 1.7rem; font-weight: 700; color: #1e293b; margin-bottom: 0.15rem; }
    .block-sub { color: #475569; margin-bottom: 0.8rem; }
    .domain-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1d4ed8;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
        color: #0f172a;
    }
    .score-pill {
        display: inline-block;
        min-width: 2.1rem;
        text-align: center;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.15rem 0.45rem;
    }
    .tag {
        display: inline-block;
        margin: 0.15rem 0.25rem 0.15rem 0;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.78rem;
        background: #e2e8f0;
        color: #1e293b;
    }
    .note-amber {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        color: #78350f;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .note-blue {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .note-red {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #7f1d1d;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_loader() -> DatasetLoader:
    loader = DatasetLoader(config=get_config())
    loader.load()
    return loader


@st.cache_resource
def get_router() -> GroqRouter:
    return GroqRouter(config=get_config().groq)


def get_llm() -> LlmService:
    return LlmService(get_router())


def _score_color(score: int) -> str:
    return {5: "#15803d", 4: "#1d4ed8", 3: "#b45309", 2: "#c2410c", 1: "#b91c1c"}.get(score, "#334155")


def _note(kind: str, text: str) -> None:
    st.markdown(f'<div class="note-{kind}">{text}</div>', unsafe_allow_html=True)


def _table(frame: pd.DataFrame) -> None:
    """Show a table without mixed-type Arrow errors."""
    shown = frame.copy()
    for column in shown.columns:
        if shown[column].dtype == object:
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else str(value))
    st.dataframe(shown, width="stretch", hide_index=True)


def render_domain_card(domain: RankedDomain) -> None:
    color = _score_color(domain.relevance_score)
    crosswalk = domain.onet_work_style_crosswalk or "No direct O*NET work-style crosswalk"
    impact = (
        f"{domain.onet_impact_score:.0f} / 100"
        if domain.onet_impact_score is not None
        else "Not in the dataset"
    )
    st.markdown(
        f"""
        <div class="domain-card">
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
            <div>
              <strong>{domain.domain_id} — {domain.domain_name}</strong>
              <div style="margin-top:0.35rem;">
                <span class="tag">{domain.construct_family}</span>
                <span class="tag">Importance: {domain.score_interpretation}</span>
                <span class="tag">Rank {domain.rank} (ties share this rank)</span>
              </div>
            </div>
            <div style="text-align:right;">
              <span class="score-pill" style="background:{color};">{domain.relevance_score}</span>
              <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">Stored 1–5 estimate</div>
            </div>
          </div>
          <p style="margin:0.55rem 0 0.2rem 0;">{domain.operational_definition}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Evidence and limitations — {domain.domain_id}"):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Why this score is in the dataset:** {domain.rationale}")
            st.markdown(f"**Dataset confidence note:** {domain.confidence}")
            st.markdown(f"**Score status:** {domain.score_status}")
            st.markdown(f"**Evidence status:** {domain.evidence_status}")
            st.markdown(f"**Review priority:** {domain.review_priority}")
            if domain.score_review_note:
                st.markdown(f"**Review note:** {domain.score_review_note}")
        with right:
            st.markdown(f"**O*NET occupation:** {domain.onet_reference_occupation}")
            st.markdown(f"**SOC:** {domain.onet_reference_soc}")
            st.markdown(f"**Profile match:** {domain.onet_profile_match_quality}")
            st.markdown(f"**Work-style crosswalk:** {crosswalk}")
            st.markdown(f"**Crosswalk quality:** {domain.crosswalk_quality}")
            st.markdown(f"**O*NET impact (0–100, separate from the 1–5 score):** {impact}")
            if domain.onet_source_url:
                st.markdown(f"[O*NET source]({domain.onet_source_url})")
        if domain.evidence_alignment_review:
            st.caption(domain.evidence_alignment_review)
        if domain.notes:
            st.caption(domain.notes)


def _store_mapping(title: str, roles: list[str], loader: DatasetLoader, top_n: int, job_text: str) -> None:
    match = match_role(title, roles)
    result = rank_domains(match.matched_role, loader, top_n=top_n) if match.matched_role else None
    st.session_state["mapping_view"] = {
        "match": match,
        "result": result,
        "job_text": job_text.strip(),
        "top_n": top_n,
    }
    st.session_state["mapping_explanation"] = None
    st.session_state["mapping_jd"] = None


def tab_role_mapping() -> None:
    st.markdown('<div class="block-title">Job-role mapping</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">Look up stored domain relevance for a reference role. '
        "Unknown titles are labeled as proxies.</div>",
        unsafe_allow_html=True,
    )
    loader = get_loader()
    roles = loader.get_roles()

    pending = st.session_state.pop("pending_title", None)
    if pending:
        _store_mapping(
            pending,
            roles,
            loader,
            int(st.session_state.get("top_n_slider", 10)),
            st.session_state.get("jd_input_mapping", ""),
        )

    left, right = st.columns([2, 1])
    with left:
        user_title = st.text_input(
            "Job title",
            placeholder="Data Scientist, ML Engineer, Product Manager…",
            key="job_title_input",
        )
    with right:
        selected = st.selectbox("Reference role", [""] + roles, key="role_dropdown")
    job_text = st.text_area(
        "Optional job description",
        height=110,
        key="jd_input_mapping",
        placeholder="Used only if you ask for a model note. It does not change stored scores.",
        help="The description is sent to Groq only when you request a model note. It is not saved in the review log.",
    )
    top_n = st.slider("How many top domains", 5, 50, 10, key="top_n_slider")
    st.caption("Domains that tie at the cutoff are included, so you may see more than this number.")

    if st.button("Show stored domains", type="primary"):
        title = user_title.strip() or selected
        if not title:
            st.warning("Enter a job title or choose a reference role.")
        else:
            _store_mapping(title, roles, loader, top_n, job_text)

    view = st.session_state.get("mapping_view")
    if not view:
        _note("blue", PROVISIONAL_NOTE)
        return

    match: RoleMatch = view["match"]
    result: RankingResult | None = view["result"]
    if match.match_type == MatchType.EXACT:
        st.success(f"Exact reference role: {match.matched_role}")
    elif match.match_type == MatchType.ALIAS:
        st.info(f"Alias proxy: “{match.input_title}” is shown with stored scores for {match.matched_role}.")
        _note("amber", match.proxy_warning or "")
    elif match.match_type == MatchType.FUZZY:
        st.warning(
            f"Fuzzy proxy: “{match.input_title}” is not a stored role. "
            f"Closest stored role is {match.matched_role} ({match.confidence:.0%} title similarity)."
        )
        _note("amber", match.proxy_warning or "")
    else:
        st.error(f"“{match.input_title}” is not a stored reference role.")
        _note("amber", match.proxy_warning or "Choose one of the closest stored roles. It will be labeled as a proxy.")
        for alt in match.alternatives:
            if st.button(f"Use {alt} as a proxy", key=f"proxy_{alt}"):
                st.session_state["pending_title"] = alt
                st.rerun()
        return

    if result is None:
        st.error("That role is not in the dataset.")
        return

    _note("blue", f"<strong>Role scope.</strong> {result.role_scope}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Domains shown", result.returned_count)
    c2.metric("Requested", result.requested_top_n)
    c3.metric("Domains for this role", result.total_domains)
    c4.metric("Tie at cutoff", "Included" if result.tie_included else "No extra tie")
    if result.tie_included:
        _note(
            "blue",
            f"Showing {result.returned_count} domains because more than one domain shares the score at the top-{result.requested_top_n} cutoff.",
        )
    _note("red", PROVISIONAL_NOTE)

    for domain in result.domains:
        render_domain_card(domain)

    export = domains_to_dataframe(result.domains)
    st.download_button(
        "Download these rows (CSV)",
        export.to_csv(index=False),
        file_name=f"{result.role.replace(' ', '_')}_top_domains.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("**Optional model note**")
    st.caption(
        "The model may explain the stored rows or comment on a job description. "
        "It cannot replace the 1–5 scores."
    )
    if not get_router().config.is_configured:
        st.info("Add GROQ_API_KEY to use model notes. The list above does not need it.")
        return

    question = st.text_input(
        "Optional question for the model",
        key="explain_question",
        placeholder="Which of these domains does the description emphasize?",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        ask = st.button("Explain the stored ranking")
    with col_b:
        ask_jd = st.button("Comment on the job description")

    if ask or ask_jd:
        with st.spinner("Calling Groq…"):
            service = get_llm()
            known = {item["domain_id"] for item in loader.get_domains()}
            if ask_jd:
                highlights = "\n".join(
                    f"{item.domain_id} {item.domain_name}: stored score {item.relevance_score}"
                    for item in result.domains[:12]
                )
                st.session_state["mapping_jd"] = service.analyze_job_description(
                    view.get("job_text") or job_text,
                    loader.get_domains(),
                    roles,
                    known,
                    reference_role=result.role,
                    stored_highlights=highlights,
                )
                st.session_state["mapping_explanation"] = None
            else:
                st.session_state["mapping_explanation"] = service.explain_ranking(
                    result, known, user_question=question or None
                )

    if st.session_state.pop("mapping_explain_retry", False):
        known_ids = {item["domain_id"] for item in loader.get_domains()}
        st.session_state["mapping_explanation"] = get_llm().explain_ranking(
            result,
            known_ids,
            user_question=question or None,
        )

    explanation = st.session_state.get("mapping_explanation")
    if explanation is not None:
        if explanation.ok:
            st.markdown(f"Model: `{explanation.model_id}` · {explanation.latency_ms:.0f} ms · attempts {explanation.attempts}")
            if explanation.fallback_used:
                st.warning("A fallback model answered. This does not bypass an account-wide quota.")
            st.markdown(explanation.text)
        else:
            st.error(explanation.error)
            if st.button("Retry explanation", key="retry_explain"):
                st.session_state["mapping_explain_retry"] = True
                st.rerun()
        for warning in explanation.warnings:
            st.warning(warning)
        st.caption("Model text is an explanation of stored rows, not a new measurement.")

    if st.session_state.pop("mapping_jd_retry", False) and result is not None:
        highlights = "\n".join(
            f"{item.domain_id} {item.domain_name}: stored score {item.relevance_score}"
            for item in result.domains[:12]
        )
        st.session_state["mapping_jd"] = get_llm().analyze_job_description(
            view.get("job_text") or "",
            loader.get_domains(),
            roles,
            {item["domain_id"] for item in loader.get_domains()},
            reference_role=result.role,
            stored_highlights=highlights,
        )
    jd_result = st.session_state.get("mapping_jd")
    if jd_result is not None:
        _render_jd_result(jd_result, roles, loader, "mapjd", "mapping_jd_retry")


def tab_full_profile() -> None:
    st.markdown('<div class="block-title">Full role profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">All stored domains for one role, sorted by the 1–5 score.</div>',
        unsafe_allow_html=True,
    )
    loader = get_loader()
    role = st.selectbox("Reference role", loader.get_roles(), key="profile_role")
    result = get_full_profile(role, loader)
    if result is None:
        st.error("Role not found.")
        return
    _note("blue", f"<strong>Role scope.</strong> {result.role_scope}")

    with st.expander("Filters", expanded=True):
        families = ["All"] + loader.get_construct_families()
        family = st.selectbox("Domain family", families, key="profile_family")
        scores = st.multiselect(
            "Relevance scores",
            [5, 4, 3, 2, 1],
            default=[5, 4, 3, 2, 1],
            key="profile_scores",
        )
        evidence_values = ["All"] + sorted({item.evidence_status for item in result.domains})
        review_values = ["All"] + sorted({item.review_priority for item in result.domains})
        evidence = st.selectbox("Evidence status", evidence_values, key="profile_evidence")
        review = st.selectbox("Review priority", review_values, key="profile_review")

    filtered = filter_domains(
        result,
        construct_family=None if family == "All" else family,
        evidence_status=None if evidence == "All" else evidence,
        review_priority=None if review == "All" else review,
    )
    allowed = set(scores) if scores else {1, 2, 3, 4, 5}
    filtered = [item for item in filtered if item.relevance_score in allowed]
    st.caption(f"{len(filtered)} of {result.total_domains} domains")
    if filtered:
        counts = pd.Series([item.relevance_score for item in filtered]).value_counts().sort_index()
        st.bar_chart(counts, x_label="Stored relevance score", y_label="Domains")
    for domain in filtered:
        render_domain_card(domain)

    frame = domains_to_dataframe(filtered)
    st.download_button(
        "Download this profile (CSV)",
        frame.to_csv(index=False),
        file_name=f"{role.replace(' ', '_')}_profile.csv",
        mime="text/csv",
        key="export_profile",
    )
    _note("red", PROVISIONAL_NOTE)

    st.markdown("**Local review note**")
    st.caption("Saved only on this computer. It does not change the dataset.")
    if filtered:
        chosen = st.selectbox(
            "Domain",
            [f"{item.domain_id} — {item.domain_name}" for item in filtered],
            key="flag_domain",
        )
        note = st.text_input("Note for an expert reviewer", key="flag_note", max_chars=500)
        if st.button("Save review note"):
            add_flag(role, chosen.split("—")[0].strip(), note)
            st.success("Saved locally.")


def tab_domain_explorer() -> None:
    st.markdown('<div class="block-title">Domain explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">Definitions come from the reference dataset. '
        "Domains in the same family, or linked to the same O*NET work style, can overlap.</div>",
        unsafe_allow_html=True,
    )
    loader = get_loader()
    domains = loader.get_domains()
    crosswalk = loader.get_reference_pack().sheet(CROSSWALK_SHEET)
    families = ["All"] + sorted({item["construct_family"] for item in domains})
    family = st.selectbox("Construct family", families, key="explorer_family")
    query = st.text_input("Search name or definition", key="domain_search")
    filtered = domains if family == "All" else [item for item in domains if item["construct_family"] == family]
    if query:
        needle = query.lower()
        filtered = [
            item
            for item in filtered
            if needle in item["domain_name"].lower() or needle in item["operational_definition"].lower()
        ]
    st.caption(f"{len(filtered)} domains")
    _note(
        "amber",
        "These 50 domains are a reference list, not 50 statistically independent factors. "
        "Shared families and shared O*NET work-style links are overlap, not extra evidence.",
    )

    style_groups: dict[str, list[str]] = {}
    if not crosswalk.empty and "O*NET_work_style" in crosswalk.columns:
        for _, row in crosswalk.iterrows():
            style = str(row.get("O*NET_work_style") or "").strip()
            if style and style.lower() != "nan":
                style_groups.setdefault(style, []).append(f"{row['domain_id']} {row['domain_name']}")

    for item in filtered:
        st.markdown(
            f"""
            <div class="domain-card">
              <strong>{item['domain_id']} — {item['domain_name']}</strong>
              <span class="tag">{item['construct_family']}</span>
              <p style="margin:0.45rem 0 0 0;">{item['operational_definition']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Occupational references — {item['domain_id']}"):
            if not crosswalk.empty:
                rows = crosswalk[crosswalk["domain_id"] == item["domain_id"]]
                if rows.empty:
                    st.caption("No row in the workbook domain-crosswalk sheet for this domain.")
                else:
                    _table(rows)
                    style = str(rows.iloc[0].get("O*NET_work_style") or "")
                    related = [name for name in style_groups.get(style, []) if not name.startswith(item["domain_id"])]
                    if related:
                        st.markdown("**Other project domains linked to the same O*NET work style:**")
                        for name in related:
                            st.markdown(f"- {name}")
            role_rows = loader.get_domain_roles(item["domain_id"])
            chart = (
                role_rows[["role", "project_relevance_score_1_to_5"]]
                .sort_values("project_relevance_score_1_to_5", ascending=False)
                .set_index("role")
            )
            st.bar_chart(chart, y="project_relevance_score_1_to_5")
            st.caption("Bars are stored 1–5 estimates for each reference role, not O*NET impact scores.")


def _render_jd_result(
    jd_result,
    roles: list[str],
    loader: DatasetLoader,
    key_prefix: str,
    retry_flag: str,
) -> None:
    _note("amber", jd_result.disclaimer)
    if not jd_result.ok and jd_result.error:
        st.error(jd_result.error)
        if st.button("Retry job-description analysis", key=f"{key_prefix}_retry"):
            st.session_state[retry_flag] = True
            st.rerun()
        if jd_result.raw_text:
            with st.expander("Unparsed model text"):
                st.write(jd_result.raw_text)
        return
    if jd_result.model_id:
        st.caption(
            f"Model `{jd_result.model_id}` · {jd_result.latency_ms:.0f} ms · "
            f"{'fallback' if jd_result.fallback_used else 'preferred model'}"
        )
    if jd_result.ambiguity_notes:
        _note("amber", f"<strong>Why the role may be ambiguous.</strong> {jd_result.ambiguity_notes}")
    if jd_result.suggested_reference_roles:
        st.markdown("**Reference roles the model suggested**")
        st.caption("Suggestions only. Choose one below to see that role’s stored scores.")
        choice = st.selectbox(
            "Stored reference role",
            ["(do not load stored scores)"] + roles,
            key=f"{key_prefix}_role",
        )
        suggested = ", ".join(jd_result.suggested_reference_roles)
        st.caption(f"Model list: {suggested}")
        if choice != "(do not load stored scores)":
            stored = rank_domains(choice, loader, top_n=8)
            if stored:
                st.markdown(f"Stored scores for **{choice}** (dataset, not the model)")
                for domain in stored.domains:
                    render_domain_card(domain)
    for heading, rows in (
        ("Primary domains mentioned by the model", jd_result.primary_domains),
        ("Other domains mentioned by the model", jd_result.secondary_domains),
    ):
        if not rows:
            continue
        st.markdown(f"**{heading}**")
        for item in rows:
            st.markdown(
                f"<div class='domain-card'><strong>{item.get('domain_id', '')} — {item.get('domain_name', '')}</strong>"
                f"<p>{item.get('reasoning', '')}</p></div>",
                unsafe_allow_html=True,
            )
    if jd_result.review_notes:
        _note("blue", f"<strong>Human review.</strong> {jd_result.review_notes}")
    if jd_result.unverified_suggestions:
        st.markdown("**Unverified suggestions**")
        st.caption("These are not reference domains and have no stored score.")
        for item in jd_result.unverified_suggestions:
            st.markdown(f"- {item}")
    for warning in jd_result.warnings:
        st.warning(warning)


def tab_jd_analysis() -> None:
    st.markdown('<div class="block-title">Job-description analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">The model maps duties to existing domain definitions. '
        "Results stay separate from the fixed reference scores.</div>",
        unsafe_allow_html=True,
    )
    loader = get_loader()
    roles = loader.get_roles()
    if not get_router().config.is_configured:
        st.warning("This page needs GROQ_API_KEY. Role mapping, profiles, and the domain explorer work without it.")
        return
    reference = st.selectbox("Optional reference role", ["(let the model suggest)"] + roles, key="jd_reference_role")
    description = st.text_area("Job description", height=240, key="jd_analysis_text")
    run = st.button("Analyze description", type="primary") or st.session_state.pop("jd_page_retry", False)
    if run:
        if not description.strip():
            st.warning("Paste a job description first.")
        else:
            highlights = None
            chosen = None if reference.startswith("(") else reference
            if chosen:
                stored = rank_domains(chosen, loader, top_n=10)
                if stored:
                    highlights = "\n".join(
                        f"{item.domain_id} {item.domain_name}: stored {item.relevance_score}"
                        for item in stored.domains
                    )
            with st.spinner("Calling Groq…"):
                st.session_state["jd_page_result"] = get_llm().analyze_job_description(
                    description,
                    loader.get_domains(),
                    roles,
                    {item["domain_id"] for item in loader.get_domains()},
                    reference_role=chosen,
                    stored_highlights=highlights,
                )
    current = st.session_state.get("jd_page_result")
    if current is not None:
        _render_jd_result(current, roles, loader, "pagejd", "jd_page_retry")


def tab_evidence() -> None:
    st.markdown('<div class="block-title">Dataset and evidence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">Structural checks are separate from occupational evidence. '
        "Nothing on this page is a predictive-validity result.</div>",
        unsafe_allow_html=True,
    )
    loader = get_loader()
    info = loader.get_dataset_info()
    pack = loader.get_reference_pack()
    recon = pack.reconciliation

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Roles", info.num_roles)
    c2.metric("Domains", info.num_domains)
    c3.metric("Mappings", info.num_mappings)
    c4.metric("Duplicate pairs", info.duplicate_role_domain)
    st.markdown(f"**CSV:** `{info.csv_path}`")
    st.markdown(f"**CSV SHA-256:** `{info.csv_sha256}`")
    st.markdown(f"**Workbook SHA-256:** `{info.xlsx_sha256}`")
    if info.csv_xlsx_scores_agree:
        st.success(info.reconciliation_summary)
    else:
        st.error(info.reconciliation_summary)
    if info.xlsx_reader_note:
        _note("amber", info.xlsx_reader_note)

    if recon.mappings:
        st.markdown(
            f"Mapping sheet rows: {recon.mappings.rows}. "
            f"Matched to CSV: {recon.mappings.matched_rows}. "
            f"Fields that differ: {recon.mappings.differing_fields or 'none'}."
        )
    for extra in recon.supplemental:
        st.markdown(
            f"- **{extra.sheet_name}:** {extra.rows} rows, matched {extra.matched_rows}, "
            f"differences {extra.differing_fields or 'none'}. {extra.note}"
        )

    st.markdown("**Score distribution (stored 1–5 estimates)**")
    score_frame = pd.DataFrame(
        {"count": pd.Series(info.score_distribution)}
    )
    st.bar_chart(score_frame)

    st.markdown("**Missing values**")
    if info.missing_values:
        for column, count in sorted(info.missing_values.items(), key=lambda pair: -pair[1]):
            st.markdown(f"- `{column}`: {count} ({count / info.num_mappings:.1%})")
    else:
        st.markdown("No missing values.")
    _note(
        "blue",
        "Empty O*NET crosswalk or impact cells mean the dataset has no source benchmark for that row. "
        "They are not filled in by this application, and they are not turned into 1–5 scores.",
    )

    st.markdown("**Evidence label on every row**")
    for status in info.evidence_statuses:
        st.markdown(f"- {status}")
    st.markdown("**Review priority counts**")
    review_counts = loader.df["review_priority"].value_counts()
    for label, count in review_counts.items():
        st.markdown(f"- {label}: {count}")

    st.markdown("**Broad or weak occupational proxies**")
    if info.weak_proxy_roles:
        st.markdown(", ".join(info.weak_proxy_roles))
    audit = pack.sheet(AUDIT_SHEET)
    if not audit.empty:
        _table(audit)
        st.caption(
            "The column numeric_O*NET_scores_applied_to_project_1_to_5 records whether "
            "O*NET numbers were copied into the project score. In this workbook the value is No."
        )

    scoring = pack.sheet(SCORING_SHEET)
    if not scoring.empty:
        st.markdown("**Scoring method (from the workbook)**")
        _table(scoring)
    qa = pack.sheet(QA_SHEET)
    if not qa.empty:
        st.markdown("**QA summary shipped in the workbook**")
        _table(qa)
        st.caption("This is the dataset author’s structural QA table, not an external validation study.")

    queue = pack.sheet(REVIEW_QUEUE_SHEET)
    if not queue.empty:
        st.markdown(f"**Expert review queue ({len(queue)} rows in the workbook)**")
        reason_col = "review_reason" if "review_reason" in queue.columns else None
        if reason_col:
            _table(queue[reason_col].value_counts().rename("rows").reset_index())
        st.download_button(
            "Download expert review queue (CSV)",
            queue.to_csv(index=False),
            file_name="expert_review_queue.csv",
            mime="text/csv",
        )
    benchmarks = pack.sheet(BENCHMARKS_SHEET)
    if not benchmarks.empty:
        st.markdown(f"**O*NET work-style benchmarks ({len(benchmarks)} rows)**")
        st.caption("0–100 source metrics. Kept separate from project relevance scores.")
        _table(benchmarks.head(25))

    st.markdown("**Notes saved in this app**")
    flags = list_flags()
    if flags:
        _table(pd.DataFrame([flag.__dict__ for flag in flags]))
        if st.button("Delete local review notes"):
            clear_flags()
            st.rerun()
    else:
        st.caption("No local review notes yet. Add one from Full role profile.")


def tab_model_status() -> None:
    st.markdown('<div class="block-title">Model status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">Preferences come from environment variables. '
        "The live catalogue decides which of those IDs can be called.</div>",
        unsafe_allow_html=True,
    )
    router = get_router()
    config = router.config
    if not config.is_configured:
        st.error("GROQ_API_KEY is not set.")
        st.markdown(
            "1. Create a key at [console.groq.com](https://console.groq.com).\n"
            "2. Copy `.env.example` to `.env`.\n"
            "3. Set `GROQ_API_KEY`.\n"
            "4. Restart the app.\n\n"
            "Role mapping still works without this key."
        )
        return

    st.success("A Groq API key is set. The key is not shown.")
    if st.button("Refresh model catalogue"):
        router.fetch_catalog(force=True)
    elif not router.get_status().catalog_fetched:
        router.fetch_catalog(force=False)

    status = router.get_status()
    if status.current_model:
        st.info(f"Last successful model: `{status.current_model}`")
    else:
        st.caption("No successful model call in this session yet.")
    if status.fallback_occurred:
        st.warning("A fallback model was used. That does not bypass an account-wide quota.")
    if status.pinned_model:
        st.caption(f"Pinned model: `{status.pinned_model}`")
    if status.catalog_age_seconds is not None:
        st.caption(f"Catalogue age: {status.catalog_age_seconds:.0f}s (cache TTL {config.catalog_ttl_seconds}s)")
    if status.last_error:
        st.error(status.last_error)
    else:
        st.caption("No inference error stored.")

    st.markdown("**Configured preference order**")
    st.code(", ".join(config.model_preference) or "(empty)")
    st.markdown("**Configured fallbacks**")
    st.code(", ".join(config.fallback_models) or "(empty)")
    if status.eligible_models:
        st.success("Eligible text models: " + ", ".join(f"`{item}`" for item in status.eligible_models))
    else:
        st.error(
            "None of the configured IDs are usable text-chat models in the current catalogue. "
            "Set GROQ_MODEL_PREFERENCE to an ID from the list below and restart."
        )
    if status.skipped_models:
        st.markdown("**Configured but not eligible:** " + ", ".join(f"`{item}`" for item in status.skipped_models))

    if "auto_routing" not in st.session_state:
        st.session_state["auto_routing"] = status.pinned_model is None
    use_auto = st.toggle("Automatic routing", key="auto_routing")
    if use_auto:
        router.unpin_model()
    elif status.eligible_models:
        pinned = st.selectbox("Pin one model", status.eligible_models, key="pin_choice")
        router.pin_model(pinned)
    elif status.available_models:
        st.caption("Pinning needs at least one eligible text model.")

    if status.available_models:
        eligible = set(status.eligible_models)
        rows = []
        for model_id in status.available_models:
            rows.append(
                {
                    "model_id": model_id,
                    "in_preference_list": model_id in set(config.all_candidate_models),
                    "eligible_for_text": model_id in eligible,
                }
            )
        _table(pd.DataFrame(rows))
    else:
        st.warning("Catalogue is empty. Check the key, network, and then refresh.")

    st.markdown("**Health prompt**")
    st.caption("Sends one short sentence. It does not include a job description.")
    if st.button("Send health prompt"):
        try:
            result = router.chat_completion(
                [{"role": "user", "content": "Reply with the single word OK."}],
                temperature=0,
                max_tokens=16,
            )
            st.success(f"{result.model_id} replied. Fallback: {result.fallback_used}. Attempts: {result.attempts}.")
            st.write(result.content)
        except Exception as exc:
            st.error(str(exc))

    st.markdown(
        """
        | Variable | What to change |
        | --- | --- |
        | `GROQ_MODEL_PREFERENCE` | First model to try, then the next choices |
        | `GROQ_FALLBACK_MODELS` | Extra IDs used after the preference list |
        | `GROQ_MAX_RETRIES` | Attempts on the same model for timeouts, HTTP 429, and HTTP 5xx |
        | `GROQ_MAX_FALLBACKS` | How many extra models may be tried after the first |
        | `GROQ_MODEL_CATALOG_TTL_SECONDS` | How long the model list is cached |
        | `GROQ_EXCLUDED_MODEL_PATTERNS` | IDs containing these words are not used for text |
        """
    )


def _opening_message(roles: list[str]) -> str:
    """First chat message: the roles a person can enter."""
    listed = "\n".join(f"{index}. {role}" for index, role in enumerate(roles, start=1))
    return (
        f"You can enter these {len(roles)} roles:\n\n"
        f"{listed}\n\n"
        "Type one of them. I will list the stored domains that fit, "
        "what each domain means, and why it fits that role."
    )


def _render_chat() -> None:
    """Type a message. The reply reads like a chat and keeps stored scores fixed."""
    st.markdown(
        """
        <style>
        .block-container { max-width: 760px; padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="block-title">Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">Type a role. Each domain includes what it means, why it fits, and its score.</div>',
        unsafe_allow_html=True,
    )
    roles = get_loader().get_roles()
    opening = _opening_message(roles)
    if "chat_messages" not in st.session_state or not st.session_state.chat_messages:
        st.session_state.chat_messages = [{"role": "assistant", "content": opening}]
    elif st.session_state.chat_messages[0].get("role") == "assistant":
        st.session_state.chat_messages[0]["content"] = opening
    if "chat_role" not in st.session_state:
        st.session_state.chat_role = None

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for warning in message.get("warnings", []):
                st.caption(warning)

    text = st.chat_input("Type a role")
    if not text:
        return

    st.session_state.chat_messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    history = st.session_state.chat_messages[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, context, warnings = reply_to_chat(
                    text,
                    get_loader(),
                    get_router(),
                    history,
                    previous_role=st.session_state.chat_role,
                )
            except Exception:
                answer = "That role could not be answered just now. Type it again."
                context = None
                warnings = []
        st.markdown(answer)
        for warning in warnings:
            st.caption(warning)
    if context is not None and context.matched_role:
        st.session_state.chat_role = context.matched_role
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": answer, "warnings": warnings}
    )


def main() -> None:
    with st.sidebar:
        st.markdown("**Psychometric domain mapping**")
        router = get_router()
        if router.config.is_configured:
            st.success("Groq key is set")
        else:
            st.info("Groq key not set. The chat still lists stored domains.")
        try:
            info = get_loader().get_dataset_info()
            st.metric("Roles", info.num_roles)
            st.metric("Domains", info.num_domains)
            st.metric("Mappings", info.num_mappings)
        except DataLoadError as exc:
            st.error(str(exc))
        st.caption("Scores are provisional role-level estimates.")
        if st.button("Clear chat"):
            st.session_state.pop("chat_messages", None)
            st.session_state.chat_role = None
            st.rerun()

    _render_chat()


main()
