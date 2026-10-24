"""The website. Type a job title, see the top domains for that job."""
import os
import streamlit as st

# Streamlit Cloud: copy Secrets into env vars (locally, .env is used instead).
try:
    for _k in ("GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_API_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from rank_domains import analyze, ranked_domains, resolve_title

st.set_page_config(page_title="Psychometric Domain Finder", layout="centered")
st.title("💬 Psychometric Domain Finder")
st.caption("Tell me a job title, and I'll tell you the psychometric domains that matter most for that role.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
        "content": "Hi! 👋 Enter any job title — for example **Data Scientist**, **Software Developer**, or **HR Manager** — and I'll share the psychometric domains most relevant to that role."}]
if "pending" not in st.session_state: st.session_state.pending = None

def render_result(resp):
    """Domain cards with score bars."""
    items = ranked_domains(resp)
    occ = resp.resolved_occupation.official_occupation_title
    st.markdown(f"Here are the top psychometric domains for **{occ}**:")
    if not items:
        st.write("No strong domain matches stood out for this role. Try another job title.")
        return
    for it in items:
        st.markdown(f"**{it['rank']}. {it['name']}** — {it['score']}% match")
        st.progress(it["score"] / 100)
        st.markdown(f"*What it means:* {it['what_it_means']}")
        if it["why_it_fits"]:
            st.markdown(f"*Why it fits:* {it['why_it_fits']}")
        st.markdown("---")
    st.write("Want to check another role? Just type its job title.")

# Render the FULL history: text messages and domain-card answers, in order.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if "result" in m:
            render_result(m["result"])
        else:
            st.markdown(m["content"])

if st.session_state.pending:
    cands = st.session_state.pending
    labels = {c.official_occupation_title: c.onet_soc_code for c in cands}
    pick = st.selectbox("Which one did you mean?", list(labels), key="pick")
    if st.button("Show domains", type="primary"):
        with st.spinner("Thinking…"):
            resp = analyze(onet_soc_code=labels[pick], top_per_family=5)
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": pick})
        st.session_state.messages.append({"role": "assistant", "result": resp})
        st.rerun()

if prompt := st.chat_input("Enter a job title…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    res = resolve_title(prompt)
    if res.status == "resolved" and res.candidates:
        with st.spinner("Thinking…"):
            resp = analyze(job_title=prompt, onet_soc_code=res.candidates[0].onet_soc_code, top_per_family=5)
        st.session_state.messages.append({"role": "assistant", "result": resp})
    elif res.candidates:
        st.session_state.pending = res.candidates
        st.session_state.messages.append({"role": "assistant",
            "content": f"That title matches **{len(res.candidates)} different occupations**. Pick the exact one below 👇"})
    else:
        st.session_state.messages.append({"role": "assistant",
            "content": f"I couldn't find an occupation matching **\"{prompt}\"**. Try a more specific job title — e.g. *Data Scientist*, *Software Developer*, *Sales Manager*…"})
    st.rerun()
