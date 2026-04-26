# GRU SYSTEM AUDIT — ADSFinal_VisaSystem

> Senior Architecture Review · /claude Boondoggle Score · Pre-WS7 Gate
>
> Audited: 2026-04-12

---

## Section 1 — What You Built (The Honest Map)

| Workstream | Function |
|---|---|
| **WS1** | Daily Greenhouse scraper → bronze/silver data lake |
| **WS2** | DOL PERM + USCIS → company sponsorship profiles |
| **WS3** | Resume PDF/DOCX → CandidateProfile (Groq LLM + regex fallback) |
| **WS4** | Sentence-transformer embeddings → semantic job matching |
| **WS5** | Recency + visa confidence + JD scan → final ranked list |
| **WS6** | LangGraph wrapper (3 nodes) around WS3→WS4→WS5 |
| **run_pipeline.py** | CLI entry point |
| **models/** | 3 Pydantic schemas (UserInputs, JobMatch, ScoredJob) |

---

## Section 2 — Boondoggle Score

### What the Human Decided

| Decision | Where |
|---|---|
| "Job matching system for international students" | Problem framing |
| F-1 OPT as primary persona | Implied by defaults |
| 5-workstream naming architecture | Conversation |
| Which 15 companies to scrape | `Scrapers/config.py` |
| Target roles (Data Engineer, Analytics Engineer, etc.) | `Scrapers/config.py` |
| Groq + Llama 3.3 for WS3 | Specified by user |
| sentence-transformers for WS4 | Specified by user |
| LangGraph for WS6 | Specified by user |
| WS7 FastAPI / WS8 Frontend next | Stated by user |

### What Claude Proposed, Human Reviewed and Approved

> **Audit correction (2026-04-13):** These values were originally flagged as
> unilateral Claude decisions. The human confirmed they were reached through
> active discussion and consciously accepted. They are reclassified as
> **human-approved defaults**. Risk level drops for demo scope.
> For a production system with real users, the risk label returns to HIGH
> until calibration data backs the numbers.

| Decision | Location | Demo Risk | Production Risk |
|---|---|---|---|
| Scoring weights: **40/35/25** (international) | `ws5` constants | 🟢 LOW | 🔴 HIGH |
| Scoring weights: **70/30** (domestic) | `ws5` constants | 🟢 LOW | 🔴 HIGH |
| WS4 weights: **50/30/10/10** | `ws4` constants | 🟢 LOW | 🔴 HIGH |
| PERM volume bands (200+ = 0.85 base score) | `ws5` `_perm_base_score` | 🟢 LOW | 🔴 HIGH |
| 14-day active job window | `ws1` constant | 🟢 LOW | 🟡 MEDIUM |
| 12,000 char resume truncation | `ws3` constant | 🟡 MEDIUM | 🟡 MEDIUM |
| Seniority thresholds (<2yr=Entry, <5yr=Mid) | `ws3` rule-based | 🟢 LOW | 🟡 MEDIUM |
| Recency decay bands (0–3d=1.0, 60+d=0.10) | `ws5` `_recency_score` | 🟢 LOW | 🟡 MEDIUM |
| Contradiction detection thresholds | `ws5` `_detect_contradiction` | 🟢 LOW | 🟡 MEDIUM |
| JD signal priority order (positive-strong first) | `ws5` `_scan_jd` | 🟢 LOW | 🟡 MEDIUM |
| 4-char minimum for substring company match | `ws5` `_find_company` | 🟢 LOW | 🟡 MEDIUM |
| Skills taxonomy (200+ skills, 8 categories) | `ws3` `SKILLS_TAXONOMY` | 🟢 LOW | 🟢 LOW |

### Score (Corrected)

```
Human decisions (independent):          9 / 21   (43%)
Human-approved defaults (via discussion): 9 / 21  (43%)
Pure Claude decisions:                   3 / 21   (14%)

Corrected intentionality score: 86% — well-designed for a demo.

Remaining gap:
  → No /v1 problem statement formally recorded
  → No documented definition of "success" for the system
  → Weights need calibration data before production use
```

---

## Section 3 — Architecture Risks

### ~~🔴 RISK 1 — Scoring Weights Were Never Calibrated~~ *(Reclassified — see correction above)*

> These weights were discussed and consciously approved. For demo scope this risk is closed.
> It reopens at production scale.

### 🟡 RISK 1 (Revised) — Weights Are Discussed Defaults, Not Calibrated Values

```python
# ws4_job_matcher.py
W_SEMANTIC  = 0.50
W_SKILL     = 0.30
W_ROLE      = 0.10
W_LOCATION  = 0.10

# ws5_confidence_scorer.py
W_INT_MATCH   = 0.40
W_INT_VISA    = 0.35
W_INT_RECENCY = 0.25
```

These numbers decide which jobs appear at the top of a candidate's list. They were chosen by Claude with no calibration data, no user study, no A/B test. A 5-point shift in `W_INT_VISA` could push a job with a negative sponsorship signal above one with a positive signal simply because its match score is higher.

**The risk:** A candidate acts on a ranking they believe reflects reality. It reflects Claude's best guess. Those are not the same thing.

**What needs to happen:** Someone with domain knowledge — an immigration attorney, an international student advisor, or a cohort of F-1 OPT candidates — needs to validate what the weights should express. Then instrument and tune.

---

### 🔴 RISK 2 — 15 Companies, Hardcoded, Is a Product Decision Dressed as a Config

The system found 64 target-role jobs across 15 companies. 3 of those companies returned 0 results because they don't use Greenhouse. That is a 20% failure rate on the scraping layer before any matching begins.

Snowflake, Notion, and Rippling are on the list. They have 0 jobs. A candidate looking for Snowflake roles gets silence. The system does not tell them why.

**The deeper issue:** The company list is a product decision — *which employers are we surfacing?* — that was never framed as a product decision. It looks like a config file. It is actually a curation strategy with no stated criteria, no refresh cadence, and no mechanism for users to request additions.

---

### 🔴 RISK 3 — PERM Data Is Frozen and the System Doesn't Know It

```
Data/PERM/PERM_Disclosure_Data_FY2025_Q4.xlsx  ← most recent file
```

The `data_freshness_score` discounts stale filings, which is good design. But there is no mechanism to update the PERM files — no cron job, no download script, no alert when data ages past a threshold.

A company that stopped sponsoring in FY2023 still appears in results with a positive (discounted) signal. A candidate cannot distinguish between "this company sponsors and the data is fresh" and "this company used to sponsor and we're not sure anymore."

---

### 🟡 RISK 4 — WS3 Truncates Resumes With No Structural Awareness

```python
MAX_RESUME_CHARS = 12_000
if len(text) > MAX_RESUME_CHARS:
    text = text[:MAX_RESUME_CHARS]  # simple slice
```

A 4-page resume often has the most recent experience at the top and education, certifications, and older roles at the bottom. For a candidate with a long research history or many certifications, credentials are silently dropped before being sent to Groq.

**Specific failure mode:** A PhD candidate with 6 pages of publications loses all publication data. The LLM infers "Bachelor's" because that's the last degree that fit in 12,000 chars. Seniority is underestimated. Matches shift downward.

---

### 🟡 RISK 5 — WS6 Is a Sequential Wrapper, Not an Orchestrator

The current WS6 has 3 nodes:

```
validate → parse_resume → match_jobs → score_jobs → END
```

`score_jobs` calls `ws5.score()` — a black box containing JD scanning, PERM lookup, contradiction detection, recency scoring, and final ranking — all in one function call.

**The consequence:** You cannot observe intermediate state between JD scanning and PERM lookup. You cannot retry one step without rerunning the other. You cannot swap scoring strategies without touching WS5 internals. The graph looks modular. The execution is not.

The 5-agent version (SignalExtractionAgent, HistoricalEvidenceAgent, AuditorAgent) was designed but not implemented. That gap is still open.

---

### 🟡 RISK 6 — No Persistence Layer Means WS7 Has Nowhere to Put Anything

WS7 is FastAPI. FastAPI endpoints return HTTP responses. Right now results go to `print()` or a Python list.

There is no database schema, no session model, no concept of a user running multiple searches, no storage for "jobs already seen" or "jobs applied to."

When WS7 is built, the first question will be *"where does the data live?"* That answer is not in this codebase and needs to be decided before a single FastAPI route is written.

---

### 🟢 RISK 7 — Embedding Model Pings HuggingFace on Every Run

```
[INFO] HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2
```

This is a metadata check, not a model download. But it is a network dependency in the hot path of every candidate match request. In an offline environment or with HuggingFace rate-limiting, this causes a silent hang.

---

## Section 4 — What Is Structurally Sound

These are genuine design strengths. Do not change them.

| Component | Why It's Right |
|---|---|
| Bronze → Silver medallion pattern (WS1) | Correct for append-only scrape pipelines. Dedup logic is clean. |
| Groq + rule-based fallback (WS3) | Graceful degradation. The fallback is comprehensive, not a stub. |
| Pre-normalized embeddings (WS4) | Dot product = cosine similarity is a correct optimization. Index build/load separation is clean. |
| Pydantic throughout | `UserInputs → CandidateProfile → JobMatch → ScoredJob` is a well-typed pipeline. Data contracts are explicit. |
| Positive-strong before negative-strong in `_scan_jd` | The reasoning is documented and correct for the edge case it handles. |
| `_find_company` 3-tier lookup | Exact → prefix → substring with minimum-length guard is pragmatic entity resolution. |

---

## Section 5 — Decisions Required Before WS7

These are not Claude's to make.

```
1. What are the scoring weight values, and what evidence validates them?

2. What is the company coverage strategy?
   → How does a company get added or removed?
   → What happens when a company leaves Greenhouse?

3. What is the PERM data refresh cadence?
   → Who runs it? How often? What triggers a stale-data warning?

4. What is the persistence model for WS7?
   → Stateless: each request is independent, nothing stored.
   → Stateful: users, saved searches, application history.
   These are not compatible. Choosing wrong means a rewrite.

5. Is this a single-user tool or a multi-user product?
   → This answer changes the authentication model, the database
     schema, the rate-limiting strategy, and the API design entirely.

6. What does "accuracy" mean for this system?
   → How would you know if the rankings are wrong?
   → Is there a feedback loop planned?
```

---

## Section 6 — Two Paths Forward

### Path A — Ship the Demo

```
Run /v1 silent  → minimal problem statement on record
Build WS7       → stateless (no DB, no auth, resume in → results out)
Build WS8       → frontend wired to WS7
Done            → working demo
```

### Path B — Ship a Product

```
Run /v1         → answer the 6 questions above
Design          → persistence layer before touching WS7
Build WS7       → with a real data model
Result          → slower, harder, correct if going to real users
```

**The stateless vs stateful decision alone will fork the WS7 architecture into two incompatible systems. Make that call before writing a single route.**

---

*Audit produced by Gru — Architecture Command Center*
*ADSFinal_VisaSystem · Branch: Akash_0405 · 2026-04-12*
