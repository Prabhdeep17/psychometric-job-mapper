"""Chat: type a job title, get the domains back like a conversation."""
import os
import streamlit as st

try:
    for _k in ("GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_API_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from rank_domains import all_domains, analyze, ranked_domains, resolve_title

st.set_page_config(page_title="Psychometric Domain Finder", layout="centered")
st.title("🧠 Psychometric Domain Finder")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content":
        "Hi! Tell me a **job title** — for example **Data Scientist**, **Software Engineer**, or **HR Manager**.\n\n"
        "I'll show you the psychometric domains most relevant to that job with a relevance percentage and a clear explanation of why each one matters."}]
if "choices" not in st.session_state:
    st.session_state.choices = None
if "pending_description" not in st.session_state:
    st.session_state.pending_description = ""


def show_domain(it, show_details=False):
    """One domain card: name, percentage bar, explanation."""
    pct = it["estimated_match_pct"]
    if pct is None:
        st.markdown(f"**{it['name']}** — _Insufficient evidence_")
        return

    # Domain name + relevance percentage
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### {it['rank']}. {it['name']}")
    with col2:
        st.metric(label="Relevance", value=f"{pct:.1f}%", label_visibility="collapsed")

    # Progress bar
    st.progress(min(pct, 100) / 100)

    # What it means (short definition)
    st.markdown(f"📖 *{it['what_it_means']}*")

    # Why it fits this job (the main explanation)
    if it["why_it_fits"]:
        st.markdown(f"**Why it fits:** {it['why_it_fits']}")

    # Optional: show expanded details
    if show_details:
        with st.expander("More details"):
            if it["tasks"]:
                st.markdown("**Supporting tasks:**")
                for task in it["tasks"]:
                    st.markdown(f"- {task}")
            st.markdown(f"**Evidence strength:** {it['evidence_strength']}")
            st.caption(f"Family: {it['family']}")


def show_result(resp, show_all=False):
    occ = resp.resolved_occupation.official_occupation_title
    st.markdown(f"### Results for **{occ}**")

    items = all_domains(resp) if show_all else ranked_domains(resp, limit=10)

    if not any(it["estimated_match_pct"] is not None for it in items):
        st.info("Insufficient evidence to rank domains for this occupation.")
        return

    # Show details toggle
    show_details = st.checkbox("Show detailed evidence", value=False,
                                key=f"details-{id(resp)}")

    for it in items:
        if not show_all and it["estimated_match_pct"] is None:
            continue
        show_domain(it, show_details=show_details)
        st.divider()


for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("result") is not None:
            show_all = st.checkbox("Show all domains", value=False, key=f"all-{i}")
            show_result(message["result"], show_all)

if st.session_state.choices:
    labels = {c.official_occupation_title: c.onet_soc_code for c in st.session_state.choices}
    pick = st.selectbox("Which occupation did you mean?", list(labels))
    if st.button("Use this occupation", type="primary"):
        with st.spinner("Analyzing…"):
            resp = analyze(job_title=pick, job_description=st.session_state.pending_description, onet_soc_code=labels[pick])
        st.session_state.messages.append({"role": "user", "content": pick})
        st.session_state.messages.append({"role": "assistant", "result": resp})
        st.session_state.choices = None
        st.rerun()

if prompt := st.chat_input("Type a job title…"):
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    title, description = lines[0], "\n".join(lines[1:])
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_description = description
    res = resolve_title(title)
    if res.status == "resolved" and res.candidates:
        with st.spinner("Analyzing…"):
            resp = analyze(job_title=title, job_description=description, onet_soc_code=res.candidates[0].onet_soc_code)
        st.session_state.messages.append({"role": "assistant", "result": resp})
    elif res.candidates:
        st.session_state.choices = res.candidates
        names = ", ".join(c.official_occupation_title for c in res.candidates[:8])
        st.session_state.messages.append({"role": "assistant",
            "content": f"**{title}** matches more than one occupation. Pick one below.\n\n{names}"})
    else:
        st.session_state.messages.append({"role": "assistant",
            "content": f"I couldn't find an occupation for **{title}**. Try a more specific job title."})
    st.rerun()
