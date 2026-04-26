"""
pipeline/ws8_frontend.py
-------------------------
WS8: Streamlit frontend for the VisaMatch job-matching system.

Calls the WS7 FastAPI backend (http://localhost:8000) and renders:
  • Sidebar  — user inputs (visa status, roles, location, relocation flag)
  • Main     — resume uploader + Analyze button
  • Results  — ranked job cards with visa-confidence badge and reasoning

Run
---
  # Start backend first (in a separate terminal):
  /opt/anaconda3/bin/python -m uvicorn pipeline.ws7_backend:app --reload --port 8000

  # Then launch this frontend:
  /opt/anaconda3/bin/streamlit run pipeline/ws8_frontend.py
"""

import io
import os
import textwrap
import time

import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE      = os.getenv("BACKEND_URL", "https://adsfinalbackend-134643354783.europe-west1.run.app")
TIMEOUT_SHORT = 5    # health check
TIMEOUT_LONG  = 120  # analyze call (pipeline can take ~30-60 s)

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "VisaMatch — AI Job Matching",
    page_icon  = "🌐",
    layout     = "wide",
)

# ── Minimal custom CSS ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* card container */
    .job-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
        background: #fafafa;
    }
    /* badge colours */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-high     { background:#d4edda; color:#155724; }
    .badge-likely   { background:#d1ecf1; color:#0c5460; }
    .badge-unknown  { background:#fff3cd; color:#856404; }
    .badge-neg      { background:#f8d7da; color:#721c24; }
    .badge-domestic { background:#e2d9f3; color:#4a0072; }
    /* score pill */
    .score-pill {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 700;
        background: #e9ecef;
        color: #212529;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _visa_badge(signal: str, confidence: float | None, requires: bool) -> str:
    """Return an HTML badge string for the sponsorship signal."""
    if not requires:
        return '<span class="badge badge-domestic">Domestic — no visa needed</span>'

    if confidence is None:
        return '<span class="badge badge-unknown">Sponsorship: Unknown</span>'

    pct = f"{confidence * 100:.0f}%"
    sig = signal.lower()

    if sig in ("positive", "likely_positive"):
        label = "Likely Sponsors"
        cls   = "badge-high" if sig == "positive" else "badge-likely"
    elif sig in ("negative", "likely_negative"):
        label = "May Not Sponsor"
        cls   = "badge-neg"
    else:
        label = "Sponsorship Unknown"
        cls   = "badge-unknown"

    return f'<span class="badge {cls}">{label} ({pct})</span>'


def _score_color(score: float) -> str:
    if score >= 0.70:
        return "🟢"
    if score >= 0.45:
        return "🟡"
    return "🔴"


def _trend_icon(trend: str | None) -> str:
    mapping = {
        "increasing":  "📈",
        "stable":      "➡️",
        "decreasing":  "📉",
        "insufficient": "❓",
    }
    return mapping.get((trend or "").lower(), "❓")


def _check_backend() -> dict | None:
    """Returns health dict or None on failure."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=TIMEOUT_SHORT)
        if r.ok:
            return r.json()
    except requests.exceptions.ConnectionError:
        pass
    return None


def _call_analyze(
    resume_bytes: bytes,
    filename: str,
    visa_status: str,
    requires_sponsorship: bool,
    target_roles: str,
    preferred_locations: str,
    open_to_relocation: bool,
    top_n: int,
) -> dict:
    """POST /analyze. Raises on non-2xx."""
    resp = requests.post(
        f"{API_BASE}/analyze",
        files   = {"resume": (filename, io.BytesIO(resume_bytes), "application/octet-stream")},
        data    = {
            "visa_status":          visa_status,
            "requires_sponsorship": str(requires_sponsorship).lower(),
            "target_roles":         target_roles,
            "preferred_locations":  preferred_locations,
            "open_to_relocation":   str(open_to_relocation).lower(),
            "top_n":                str(top_n),
        },
        timeout = TIMEOUT_LONG,
    )
    resp.raise_for_status()
    return resp.json()


# ── Sidebar — User Inputs ──────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/earth-planet.png",
        width=64,
    )
    st.title("VisaMatch")
    st.caption("AI-powered job matching for international students")
    st.divider()

    st.subheader("Your Profile")

    visa_status = st.selectbox(
        "Visa / Work Authorization",
        options=[
            "F-1 OPT",
            "F-1 STEM OPT",
            "H-1B",
            "Green Card",
            "US Citizen",
            "Other",
        ],
        index=0,
        help="Determines which sponsorship scoring weights are applied.",
    )

    requires_sponsorship = st.toggle(
        "I need visa sponsorship",
        value=True,
        help="Turn off if you are a citizen / permanent resident.",
    )

    st.divider()
    st.subheader("Job Preferences")

    ROLE_OPTIONS = [
        "Data Engineer",
        "Analytics Engineer",
        "Data Scientist",
        "Data Analyst",
        "ML Engineer",
        "Software Engineer",
        "Backend Engineer",
        "Full Stack Engineer",
    ]
    target_roles_list = st.multiselect(
        "Target Roles",
        options   = ROLE_OPTIONS,
        default   = ["Data Engineer", "Analytics Engineer"],
    )

    LOCATION_OPTIONS = [
        "Remote",
        "New York, NY",
        "San Francisco, CA",
        "Seattle, WA",
        "Austin, TX",
        "Chicago, IL",
        "Boston, MA",
        "Los Angeles, CA",
    ]
    preferred_locations_list = st.multiselect(
        "Preferred Locations",
        options = LOCATION_OPTIONS,
        default = ["Remote"],
    )

    open_to_relocation = st.toggle("Open to relocation", value=True)

    st.divider()
    top_n = st.slider(
        "Top results to show",
        min_value = 5,
        max_value = 100,
        value     = 30,
        step      = 5,
    )

    st.divider()

    # Backend health indicator
    health = _check_backend()
    if health is None:
        st.error("Backend offline\n\nStart WS7 then refresh.")
    else:
        all_ready = (
            health.get("jobs_silver_exists")
            and health.get("sponsorship_exists")
            and health.get("index_exists")
        )
        if all_ready:
            st.success("Backend ready")
        else:
            missing = [
                k.replace("_exists", "").replace("_", " ").title()
                for k in ("jobs_silver_exists", "sponsorship_exists", "index_exists")
                if not health.get(k)
            ]
            st.warning(
                f"Data layer not ready.\nMissing: {', '.join(missing)}.\n\n"
                "Run `POST /refresh` once to build the index."
            )
        if health.get("refresh_running"):
            st.info("Data refresh in progress...")
        if health.get("last_refresh"):
            st.caption(f"Last refresh: {health['last_refresh'][:19]}")


# ── Main panel ─────────────────────────────────────────────────────────────────

st.title("Find Jobs That Work For You")
st.markdown(
    "Upload your resume and get a ranked list of matching jobs — "
    "with a **visa sponsorship confidence score** for every result."
)

col_upload, col_spacer = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type        = ["pdf", "docx", "doc"],
        label_visibility = "collapsed",
    )

    analyze_btn = st.button(
        "Analyze",
        type     = "primary",
        disabled = uploaded_file is None,
        use_container_width = True,
    )

if not uploaded_file:
    st.info("Upload a PDF or DOCX resume to get started.")
    st.stop()


# ── Run analysis ───────────────────────────────────────────────────────────────

if analyze_btn:
    # Validate sidebar inputs
    if not target_roles_list:
        st.error("Select at least one target role in the sidebar.")
        st.stop()
    if not preferred_locations_list:
        st.error("Select at least one preferred location in the sidebar.")
        st.stop()

    target_roles_str      = ",".join(target_roles_list)
    preferred_locs_str    = ",".join(preferred_locations_list)

    with st.spinner("Parsing resume and ranking jobs... (this can take ~30-60 seconds)"):
        t0 = time.time()
        try:
            data = _call_analyze(
                resume_bytes         = uploaded_file.read(),
                filename             = uploaded_file.name,
                visa_status          = visa_status,
                requires_sponsorship = requires_sponsorship,
                target_roles         = target_roles_str,
                preferred_locations  = preferred_locs_str,
                open_to_relocation   = open_to_relocation,
                top_n                = top_n,
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot reach the backend. Make sure WS7 is running:\n\n"
                "```\n"
                "/opt/anaconda3/bin/python -m uvicorn pipeline.ws7_backend:app "
                "--reload --port 8000\n"
                "```"
            )
            st.stop()
        except requests.exceptions.Timeout:
            st.error(
                "The pipeline timed out. The job index may not be built yet.\n\n"
                "Call `POST /refresh` from http://localhost:8000/docs, wait a few "
                "minutes, then try again."
            )
            st.stop()
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Backend error: {detail}")
            st.stop()

    elapsed  = time.time() - t0
    results  = data.get("results", [])
    total    = data.get("total", 0)

    st.success(f"Found **{total} matches** in {elapsed:.1f}s")

    # Store results in session state so they persist across reruns
    st.session_state["results"]           = results
    st.session_state["requires_spons"]    = requires_sponsorship


# ── Render stored results ──────────────────────────────────────────────────────

results = st.session_state.get("results", [])
if not results:
    st.stop()

req_spons = st.session_state.get("requires_spons", True)

# ── Summary metrics row ────────────────────────────────────────────────────────

n_high    = sum(1 for j in results if (j.get("sponsorship_signal") or "").lower() in ("positive", "likely_positive") and req_spons)
n_unknown = sum(1 for j in results if (j.get("sponsorship_signal") or "").lower() == "unknown" and req_spons)
n_neg     = sum(1 for j in results if (j.get("sponsorship_signal") or "").lower() in ("negative", "likely_negative") and req_spons)
avg_score = sum(j.get("final_score", 0) for j in results) / max(len(results), 1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total matches",       len(results))
m2.metric("Avg final score",     f"{avg_score:.2f}")
if req_spons:
    m3.metric("Likely sponsors",  n_high)
    m4.metric("Unknown / at risk", n_unknown + n_neg)
else:
    m3.metric("Role matches",     sum(1 for j in results if j.get("role_match")))
    m4.metric("Location matches", sum(1 for j in results if j.get("location_match")))

st.divider()

# ── Filter bar ─────────────────────────────────────────────────────────────────

f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
with f_col1:
    filter_signal = st.selectbox(
        "Filter by sponsorship signal",
        options=["All", "Likely sponsors", "Unknown", "May not sponsor"],
        index=0,
        label_visibility="collapsed",
    )
with f_col2:
    filter_role = st.checkbox("Target role only", value=False)
with f_col3:
    sort_by = st.selectbox(
        "Sort by",
        options=["Final score", "Match score", "Recency"],
        index=0,
        label_visibility="collapsed",
    )

# Apply filters
filtered = list(results)

if filter_signal != "All" and req_spons:
    sig_map = {
        "Likely sponsors":  ("positive", "likely_positive"),
        "Unknown":          ("unknown",),
        "May not sponsor":  ("negative", "likely_negative"),
    }
    keep = sig_map.get(filter_signal, ())
    filtered = [j for j in filtered if (j.get("sponsorship_signal") or "").lower() in keep]

if filter_role:
    filtered = [j for j in filtered if j.get("role_match")]

sort_key_map = {
    "Final score":  "final_score",
    "Match score":  "match_score",
    "Recency":      "recency_score",
}
skey     = sort_key_map[sort_by]
filtered = sorted(filtered, key=lambda j: j.get(skey, 0), reverse=True)

if not filtered:
    st.warning("No results match the current filters.")
    st.stop()

st.markdown(f"**Showing {len(filtered)} result(s)**")

# ── Job cards ─────────────────────────────────────────────────────────────────

for idx, job in enumerate(filtered, start=1):
    final    = job.get("final_score",       0.0)
    match    = job.get("match_score",       0.0)
    recency  = job.get("recency_score",     0.0)
    visa_c   = job.get("visa_confidence")
    signal   = job.get("sponsorship_signal", "n/a")
    trend    = job.get("sponsorship_trend", "n/a")
    perm_cnt = job.get("perm_filings_total", 0)
    days_ago = job.get("days_since_posted", -1)
    skills   = job.get("matched_skills",   [])
    reason   = job.get("reasoning",         "")
    title    = job.get("title",             "Untitled")
    company  = job.get("company_slug",      "").replace("-", " ").title()
    location = job.get("location",          "")
    apply    = job.get("apply_url",         "#")
    role_ok  = job.get("role_match",        False)
    loc_ok   = job.get("location_match",    False)

    badge_html = _visa_badge(signal, visa_c, req_spons)
    em         = _score_color(final)
    trend_ico  = _trend_icon(trend)
    posted_str = f"{days_ago}d ago" if days_ago >= 0 else "date unknown"

    with st.container():
        st.markdown(
            f"""
            <div class="job-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span style="font-size:1.12rem; font-weight:700;">{idx}. {title}</span>
                        &nbsp;&nbsp;
                        <span style="color:#555; font-size:0.95rem;">{company}</span>
                    </div>
                    <div style="text-align:right; font-size:0.85rem; color:#777;">
                        {em} <strong>{final:.2f}</strong>&nbsp;final score
                    </div>
                </div>
                <div style="font-size:0.85rem; color:#666; margin:4px 0 8px;">
                    📍 {location} &nbsp;·&nbsp; 🕐 Posted {posted_str}
                    {"&nbsp;·&nbsp; ✅ Target role" if role_ok else ""}
                    {"&nbsp;·&nbsp; 📌 Location match" if loc_ok else ""}
                </div>
                <div style="margin-bottom:8px;">
                    {badge_html}
                    <span class="badge" style="background:#f0f0f0;color:#333;">
                        Match {match*100:.0f}%
                    </span>
                    <span class="badge" style="background:#f0f0f0;color:#333;">
                        Recency {recency*100:.0f}%
                    </span>
                    {f'<span class="badge" style="background:#f0f0f0;color:#333;">PERM {perm_cnt} filings {trend_ico}</span>' if req_spons and perm_cnt else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Expandable detail panel
        with st.expander("Details & Reasoning"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Score breakdown**")
                st.progress(final,   text=f"Final score: {final:.2f}")
                st.progress(match,   text=f"Match score: {match:.2f}")
                st.progress(recency, text=f"Recency:     {recency:.2f}")
                if req_spons and visa_c is not None:
                    st.progress(visa_c, text=f"Visa confidence: {visa_c:.2f}")

            with col_b:
                st.markdown("**Matched skills**")
                if skills:
                    chips = " &nbsp;".join(
                        f'<code style="background:#eef;padding:1px 6px;border-radius:4px;">{s}</code>'
                        for s in skills[:20]
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                    if len(skills) > 20:
                        st.caption(f"… and {len(skills)-20} more")
                else:
                    st.caption("No direct skill matches detected.")

            st.markdown("**Why this job was ranked here**")
            st.info(reason or "No reasoning available.")

            st.link_button("Apply now", apply, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "VisaMatch · ADSFinal_VisaSystem · "
    "Scores are demo estimates — not legal or immigration advice."
)
