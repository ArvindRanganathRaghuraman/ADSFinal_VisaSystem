# Software Design Document
## Clocks Ticking — AI-Powered Job Matching for International Students

---

| Field       | Value                                      |
|-------------|---------------------------------------------|
| Version     | 1.0.0                                       |
| Date        | 2026-04-20                                  |
| Author      | Arvind Ranganath Raghuraman                 |
| Status      | Draft — not yet governing production        |
| Changelog   | v1.0.0 — Initial SDD from codebase audit    |

---

## 1. ONE-PAGE PROBLEM SUMMARY

Imagine you are an international student with 90 days of OPT remaining. You find a promising role — right title, right company, right city. You apply. You move through four rounds of interviews over six weeks. Then the recruiter tells you the company does not sponsor visas.

You just lost six weeks you cannot get back. The OPT clock kept running.

This is not a rare edge case. It is the default experience for international candidates relying on scattered, stale sources. The U.S. Department of Labor publishes H-1B approval records. Job boards list open roles. A few websites claim to track which companies sponsor. None of them agree with each other, and none of them tell you what is true *right now*. A company that sponsored 600 visas in 2021 may have quietly stopped in 2023. A job description may bury "must be authorized to work without sponsorship" in the fourth bullet of a twelve-point requirements list.

The deeper failure is not missing data — it is **stale data presented with false confidence**. That combination is worse than having no data at all, because it produces wrong answers that feel trustworthy.

There is a second missing piece. Even knowing which companies sponsor, students still waste weeks applying to roles they are under-qualified for, over-qualified for, or simply not aligned with. The sponsorship problem and the matching problem compound each other.

**Clock's Ticking** addresses both together. It is not a prediction model — it is a **computational skepticism pipeline** combined with a resume-aware job matching engine. The distinction matters: a prediction model asks "what is the most likely answer?" A skepticism pipeline asks "how much should I trust this answer, and what would make it wrong?"

The system answers: *"Strong PERM history through 2022, but filings have dropped sharply since 2023 and recent job postings include exclusionary sponsorship language. Confidence: Low."* — not a binary yes/no that wastes the candidate's remaining OPT window.

**This system is** a job discovery and ranking tool for international CS/data/analytics students that occupies the space between a generic job board (Indeed, LinkedIn) and a dedicated visa-tracking service. It succeeds when a candidate can upload a resume and receive a ranked, explained list of visa-sponsor-likely jobs matched to their profile — in under two minutes — without manually cross-referencing USCIS databases. The goal is not simply to be correct. The goal is to be **trustworthy** in a domain where incorrect guidance wastes irreplaceable time.

---

## 2. ARCHITECTURE PRINCIPLES

### P1 — Data Provenance Over Inference
Every sponsorship confidence score must trace back to a primary source (DOL PERM filing, USCIS petition record, or explicit JD language). No sponsorship claim is synthesized without a labeled source. A score without a documented source is a liability, not a feature.

- **Honors**: Storing `perm_by_year`, `uscis_total_approvals`, `sponsorship_signal` as separate, traceable fields.
- **Violates**: Generating a combined "sponsorship probability" from an ML model with no audit trail.
- **Failure state**: A candidate spends six weeks interviewing at a company VisaMatch scored as high-confidence; the company stopped sponsoring two years ago. The OPT clock ran the entire time. Trust in the system — and the candidate's visa timeline — collapses together.

### P2 — Graceful Degradation at Every External Boundary
Every external call (Groq LLM, HuggingFace model, job board scraping) must have a documented fallback that preserves partial functionality. A single service outage must not render the entire pipeline non-functional.

- **Honors**: Groq resume parsing falls back to rule-based NLP; `is_active` flags preserve last-known job state when scraping fails.
- **Violates**: Blocking the analysis pipeline if the embedding index hasn't been built yet.
- **Failure state**: Groq is rate-limited during a demo. The system returns nothing. User concludes the product is broken.

### P3 — Explicit Scoring, No Black Boxes
Every ranked output must carry a human-readable reasoning string that explains the score components. Weights are named constants, not magic numbers. Any change to scoring logic must update a documented constant, not an inline literal.

- **Honors**: `reasoning` field on every `ScoredJob`; named weight constants (`W_INT_MATCH = 0.40`) in `ws5_confidence_scorer.py`.
- **Violates**: Embedding score adjustments inside conditional branches without surfacing them in the output.
- **Failure state**: A hiring manager or student asks "why is this job ranked first?" and the answer is "the model said so." Product trust evaporates.

### P4 — Separation of Data Layers (Medallion Architecture)
Raw scraped data (bronze) is never mutated. Derived/cleaned data (silver) is rebuilt from bronze on each refresh. No pipeline step reads from bronze at query time. This separation makes data auditing, debugging, and re-processing deterministic.

- **Honors**: Bronze = raw JSON snapshots per date; Silver = deduplicated parquet built from all bronze.
- **Violates**: Modifying a bronze record in place to "fix" a scraping error.
- **Failure state**: A silver rebuild produces different results than expected because bronze was patched inconsistently. Debugging becomes archaeological.

---

## 3. CORE USER FLOWS + SYSTEM INTERACTION MAP

### 3.1 PRIMARY FLOW — Resume Upload → Ranked Results

```
1. User opens Streamlit UI (ws8)
   └─ Selects: visa status, sponsorship toggle, target roles, preferred locations

2. User uploads resume (PDF or DOCX)
   └─ Frontend validates: file type, roles selected, locations selected

3. User clicks "Analyze"
   └─ Frontend POSTs resume + form data to POST /analyze (ws7)
   └─ 120-second timeout

4. FastAPI backend (ws7)
   └─ Validates file type
   └─ Checks embedding index exists → 503 if not
   └─ Saves resume to temp file
   └─ Calls run_pipeline(temp_path, user_inputs, top_n=30) → ws6

5. LangGraph Pipeline (ws6) — 8 agent nodes in sequence:
   a. ScraperAgent      → Skips if silver jobs exist (unless refresh=True)
   b. SponsorshipAgent  → Skips if sponsorship parquet exists (unless refresh=True)
   c. IndexBuilderAgent → Skips if embeddings exist (unless refresh=True)
   d. ResumeParserAgent → Groq LLM extraction → CandidateProfile
   e. MatchingAgent     → Embed profile → cosine similarity → top 50 JobMatch
   f. SignalAgent       → JD text scan for sponsorship phrases (international only)
   g. EvidenceAgent     → PERM/USCIS lookup per company (international only)
   h. AuditorAgent      → Cross-validate signals, assign visa_confidence, rank, reason

6. FastAPI returns AnalyzeResponse{total, results: [ScoredJob]}
   └─ Temp file cleaned up

7. Frontend renders results
   └─ Summary metrics, filter bar, job cards
   └─ Each card: company, title, location, scores, badges, reasoning, apply link

8. User filters + clicks "Apply Now"
   └─ Redirected to Greenhouse job posting
```

**Decision Points**:
- Step 4: If embedding index missing → pipeline aborts with 503; user must trigger `/refresh` first.
- Step 5d: If Groq fails → rule-based NLP fallback; partial profile with lower confidence.
- Step 5f/5g: If `requires_sponsorship=False` → skipped entirely; domestic scoring formula used.

**Flow Honesty Test**: Built as a CLI with no UI, no branding — `run_pipeline.py --resume resume.pdf` — it solves the problem. The UI is a delivery mechanism, not a functional dependency.

---

### 3.2 INTEGRATION FLOW — System to System

| External System         | Protocol       | Data Exchanged                         | Failure Owner      |
|-------------------------|----------------|----------------------------------------|--------------------|
| Greenhouse API          | HTTP GET       | Job JSON (title, description, location, URL) | WS1 scraper       |
| Lever / Ashby           | HTTP GET       | Job JSON (ATS-specific format)         | WS1 scraper        |
| Groq LLM API            | HTTPS POST     | Resume text → structured JSON profile  | WS3 (fallback NLP) |
| HuggingFace Model Hub   | HTTPS GET      | `all-MiniLM-L6-v2` model download      | WS4 (cached after first load) |
| DOL PERM Excel Files    | Local file     | Excel sheets → company PERM history    | Manual refresh     |
| USCIS CSV               | Local file     | Tab-delimited → employer petitions     | Manual refresh     |

---

### 3.3 ADMINISTRATIVE FLOW — Operator Path

```
1. Initial setup:
   └─ Clone repo, install dependencies (uv sync or pip install -r requirements.txt)
   └─ Copy .env.example → .env, set GROQ_API_KEY
   └─ Run: python -m pipeline.ws1_run_scraper (build silver jobs)
   └─ Run: python -m pipeline.ws2_build_sponsorship (build sponsorship history)
   └─ Run: python -m pipeline.ws4_job_matcher --build (build embedding index)
   └─ Start: uvicorn pipeline.ws7_backend:app --port 8000
   └─ Start: streamlit run pipeline/ws8_frontend.py

2. Daily refresh (automated):
   └─ scripts/setup_cron.sh installs cron job (14:00 UTC / 09:00 EST)
   └─ daily_refresh.sh: WS1 → WS2 → logs to logs/daily_refresh.log

3. Adding a company:
   └─ Edit Scrapers/companies.json (add slug, display_name, ATS platform)
   └─ Trigger WS1 + WS4 rebuild

4. Updating PERM/USCIS data:
   └─ Download new Excel/CSV from DOL/USCIS portals
   └─ Place in Data/PERM/ or Data/USCIS/
   └─ Re-run WS2 (python -m pipeline.ws2_build_sponsorship)

5. Monitoring:
   └─ GET /health — checks data layer existence + refresh status
   └─ logs/daily_refresh.log — cron output
   └─ Streamlit sidebar health indicator
```

---

## 4. USER AND BUSINESS NEEDS

### User Needs

**UN1** — An F-1 OPT candidate with a finite visa window must be able to receive a ranked list of open jobs likely to offer sponsorship when they upload their resume and select their target roles — without spending hours manually cross-referencing USCIS PERM databases, and without discovering a company doesn't sponsor only after completing multiple interview rounds.

- Component: WS5 (AuditorAgent), WS6 (LangGraph), WS7 (/analyze endpoint)
- Pass/Fail: Given a resume, returns ≥1 result with `visa_confidence > 0` and a populated `reasoning` field within 120 seconds.

**UN2** — A candidate must be able to understand why a specific job was ranked where it was, when they expand a job card, without needing to interpret raw scores.

- Component: WS5 reasoning builder, WS8 job card expand view
- Pass/Fail: Every `ScoredJob` has a non-empty `reasoning` string that references at least one of: skill match, visa confidence, PERM history, JD signal.

**UN3** — A candidate must be able to filter results by sponsorship signal and role match when viewing ranked results, without losing all other result context.

- Component: WS8 filter bar
- Pass/Fail: Applying a filter reduces the displayed card count without restarting the pipeline.

**UN4** — A candidate must be able to navigate from a VisaMatch result to the original job posting when they decide to apply, without re-searching for the job externally.

- Component: WS8 "Apply Now" link, `apply_url` field on `ScoredJob`
- Pass/Fail: Every job card renders a valid Greenhouse/Lever/Ashby URL that opens the correct posting.

### Operator Needs

**ON1** — An operator must be able to add or remove target companies from the scraping list when company coverage needs to change, without modifying Python source code.

- Component: `Scrapers/companies.json`, WS1
- Pass/Fail: Adding a company to `companies.json` and re-running WS1 results in that company's jobs appearing in silver layer within one run.

**ON2** — An operator must be able to trigger a full data refresh (jobs + sponsorship + index) via the API when data becomes stale, without restarting the application.

- Component: WS7 POST /refresh endpoint
- Pass/Fail: POST /refresh returns 202; GET /health subsequently shows updated `last_refresh` timestamp.

### Business Needs

**BN1** — The system must score sponsorship confidence from verifiable primary sources (PERM filings, USCIS data) so that every ranking is defensible and auditable. The system does not provide legal immigration advice and must never present a confidence score as a guarantee — but the evidence behind each score must be traceable, because a student acting on an unsupported "high confidence" label risks wasting weeks of irreplaceable OPT time.

- Component: WS2, WS5 (source labeling), ScoredJob `reasoning` field
- Pass/Fail: Every non-zero `visa_confidence` score has a labeled source in `reasoning` (e.g., "123 PERM certifications on record").

**BN2** — The system must degrade gracefully when external services are unavailable so that a Groq outage or scraping failure does not produce a zero-result response without explanation.

- Component: WS3 fallback NLP, WS7 error responses, WS6 error routing
- Pass/Fail: When Groq is unavailable, the pipeline completes using rule-based NLP and the response includes a fallback indicator.

---

## 5. CORE COMPONENT DOCUMENTATION

---

### WS1 — Multi-ATS Job Scraper
**File**: `pipeline/ws1_run_scraper.py` + `Scrapers/Job_Scraping.py`

**Needs Served**: ON1, ON2

**Problem it solves**: Job postings across 40+ companies live on three different ATS platforms with different response schemas. The scraper normalizes them into a single bronze schema and maintains a deduplicated silver layer with job lifecycle tracking.

**How it works**:
- Input: `Scrapers/companies.json` (40 companies, ATS platform per company)
- For each company: HTTP GET to Greenhouse/Lever/Ashby endpoint, 1.2s delay, 15s timeout
- Output (Bronze): `Scrapers/data/bronze/jobs/jobs_YYYY-MM-DD.json` — raw daily snapshot
- Output (Silver): `Scrapers/data/silver/jobs_all.parquet` — deduplicated, lifecycle-tracked
- Deduplication key: `job_id`
- Active flag: `is_active = True` if `last_seen` within 14 days
- Target role flag: `is_target_role = True` if title contains Data Engineer / Analyst / Scientist keywords

**Principle Alignment**: P4 (Medallion Architecture — bronze never mutated)

**Flow Placement**: WS6 Step 1 (ScraperAgent)

**Edge Cases**:
1. Company's jobs are on a non-configured ATS → 0 jobs scraped silently; `is_active` degrades to False after 14 days.
2. Greenhouse returns 429 (rate limit) → request fails, company skipped for that run; last-known data retained.
3. Job description HTML contains malformed tags → BeautifulSoup strips best-effort; truncated description stored.

**Scope Boundary**: Does NOT embed jobs (WS4), does NOT parse job descriptions for sponsorship phrases (WS5), does NOT validate that scraped URLs are still live.

---

### WS2 — Sponsorship History Builder
**File**: `pipeline/ws2_build_sponsorship.py`

**Needs Served**: UN1, BN1

**Problem it solves**: The U.S. Department of Labor publishes H-1B LCA approval records and USCIS publishes employer petition counts — but both are raw government files with inconsistent company naming, multiple fiscal-year schemas, and no indication of whether a company's historical pattern still reflects its current intent. A company that sponsored 400 roles in FY2021 and zero in FY2024 looks identical to a first-time sponsor if you only read total counts. Students relying on stale aggregates get confidently wrong answers. WS2 normalizes these files, computes per-year trends, freshness decay, and approval-rate signals so that WS5 can detect when historical data has stopped reflecting reality — and say so explicitly instead of recycling an outdated "yes."

**How it works**:
- Input: `Data/PERM/PERM_Disclosure_Data_FY*.xlsx` (multiple fiscal years, two schemas), `Data/USCIS/Employer Information.csv`
- Company name normalization: lowercase → strip legal suffixes (Inc, LLC, Corp, Ltd) → strip punctuation
- PERM aggregation: count certified vs denied per company per fiscal year; compute trend (increasing/stable/decreasing/decreasing_sharply); compute freshness score (1.0 if filed within 1 FY, 0.1 if 4+ years old)
- USCIS aggregation: sum approvals/denials per normalized company name
- Merge PERM + USCIS on normalized name
- Output: `Data/silver/sponsorship_history.parquet` (1 row per unique company)

**Principle Alignment**: P1 (Data Provenance), P4 (Medallion Architecture)

**Flow Placement**: WS6 Step 2 (SponsorshipAgent)

**Edge Cases**:
1. Company appears in PERM but not USCIS → PERM score used; USCIS fields zeroed.
2. Company name has multiple normalized variants (e.g., "Google LLC" and "Google Inc") → treated as separate entities; only the highest-match one is found during WS5 lookup.
3. New PERM Excel file has a different column schema → schema detection logic handles two known variants; third variant causes load failure.

**Scope Boundary**: Does NOT scrape live government sites; does NOT auto-download updated PERM/USCIS files; does NOT handle non-US sponsorship data.

---

### WS3 — Resume Parser
**File**: `pipeline/ws3_resume_parser.py`

**Needs Served**: UN1, UN2

**Problem it solves**: Resumes arrive as unstructured PDFs or DOCX files. WS3 extracts structured features (skills, experience, seniority, degree) so that WS4 can embed the candidate's profile and WS5 can reason about role fit.

**How it works**:
- Input: Resume file path (PDF via pdfplumber, DOCX via python-docx) + `UserInputs`
- Truncate text to 12,000 chars (LLM context limit)
- **Primary path**: Groq API call (`llama-3.3-70b-versatile`) with structured extraction prompt → returns JSON with 15+ fields
- **Fallback**: Rule-based NLP (regex for years, skills taxonomy of 200 terms, seniority inference rules)
- Merge extracted fields with `UserInputs` (visa status, target roles, locations)
- Validate with Pydantic → return `CandidateProfile`

**Principle Alignment**: P2 (Graceful Degradation — fallback on Groq failure), P3 (Explicit scoring — seniority level is a named field, not inferred silently)

**Flow Placement**: WS6 Step 4 (ResumeParserAgent)

**Edge Cases**:
1. Resume exceeds 12,000 chars → trailing sections (education, older roles) truncated; seniority may be underestimated.
2. Groq returns malformed JSON → fallback NLP triggered; profile marked with lower confidence.
3. DOCX with embedded images/charts → python-docx extracts text only; visual-only content lost silently.

**Scope Boundary**: Does NOT parse non-English resumes, does NOT extract GPA or publication lists, does NOT handle scanned PDFs (non-text PDFs return empty text).

---

### WS4 — Semantic Job Matcher
**File**: `pipeline/ws4_job_matcher.py`

**Needs Served**: UN1, UN2

**Problem it solves**: Keyword matching cannot capture that "Data Infrastructure Engineer" is a strong match for a "Data Engineer" resume. WS4 uses sentence-transformers to embed both the candidate profile and all job descriptions, then ranks by cosine similarity blended with skill overlap and role/location flags.

**How it works**:
- **Index build** (offline step): Load silver jobs → filter active + target-role → embed each job (title + company + first 1,500 chars of description) using `all-MiniLM-L6-v2` (384-dim) → normalize → save `job_embeddings.npy` + `job_index.parquet`
- **Match** (online step): Build candidate text from skills + titles + target roles + degree → embed → cosine similarity vs all job embeddings → compute skill overlap (exact substring), role match (keyword in title), location match → blend scores → return top N `JobMatch` objects

**Scoring**:
```
match_score = 0.50 × semantic_score
            + 0.30 × skill_overlap_score
            + 0.10 × role_match (bool → 0/1)
            + 0.10 × location_match (bool → 0/1)
```

**Principle Alignment**: P3 (Explicit scoring — named weight constants), P2 (Index cached; not rebuilt on every request)

**Flow Placement**: WS6 Steps 3 (index build) and 5 (match query)

**Edge Cases**:
1. Embedding index missing when `/analyze` is called → WS7 returns 503 with instructions to trigger `/refresh`.
2. Candidate has zero extracted skills (Groq + fallback both fail) → semantic embedding still works; skill_overlap_score = 0.
3. HuggingFace model not cached → first-run network call; fails in offline environments silently.

**Scope Boundary**: Does NOT re-rank based on user feedback, does NOT update embeddings incrementally (full rebuild on each refresh), does NOT support non-English job descriptions.

---

### WS5 — Confidence Scorer & Ranker
**File**: `pipeline/ws5_confidence_scorer.py`

**Needs Served**: UN1, UN2, BN1, BN2

**Problem it solves**: A great skill match at a company that stopped sponsoring two years ago is not a good recommendation — it is a time trap. International students on OPT cannot afford to discover that mid-interview. WS5 is the skepticism layer: rather than answering "does this company sponsor?" with a binary yes/no, it asks "how confident should we be, and what would make that confidence wrong?" It combines JD language signals (job descriptions routinely bury "must be authorized to work without sponsorship" in dense requirements lists), PERM filing trends, USCIS approval history, and data freshness into a single calibrated `visa_confidence` score with a traceable plain-English explanation. When evidence conflicts — strong historical approvals but recent exclusionary JD language — it surfaces the contradiction instead of silently resolving it in either direction.

**How it works**:
- **Recency** (0–1): Tiered decay — 0–3 days = 1.00, 4–7 = 0.85, 8–14 = 0.70, 15–30 = 0.50, 31–60 = 0.30, 60+ = 0.10
- **JD Signal**: Regex cascade on job description — positive strong ("will sponsor") → +0.30; negative strong ("no sponsorship") → -0.40; positive weak → +0.15; negative weak → -0.20
- **PERM Base Score**: `max(perm_certified, uscis_approvals)` drives volume band (0 → 0.30, 200+ → 0.85); trend adjustment (increasing +0.10, decreasing −0.05, decreasing_sharply −0.15); freshness discount; clamped [0.05, 0.95]. Unknown company (not in DB) = 0.40 base.
- **Contradiction detection**: Not implemented in current codebase — planned but not built (see OQ6)
- **Final score** (international): `0.40 × match_score + 0.35 × visa_confidence + 0.25 × recency_score`
- **Final score** (domestic): `0.70 × match_score + 0.30 × recency_score`
- **Reasoning builder**: Generates plain-English summary combining all score components

**Principle Alignment**: P1 (every score traces to a source), P3 (all weights are named constants)

**Flow Placement**: WS6 Steps 6 (SignalAgent), 7 (EvidenceAgent), 8 (AuditorAgent)

**Edge Cases**:
1. Company name in job record doesn't match any entry in sponsorship history → fallback to substring match → if still no match, base score defaults to 0.30 (unknown).
2. JD contains both positive and negative sponsorship phrases → cascade logic picks highest-priority match; lower-priority phrases ignored.
3. Candidate sets `requires_sponsorship=False` → Steps 6 and 7 skipped entirely; domestic scoring formula applied.

**Scope Boundary**: Does NOT provide legal immigration advice, does NOT guarantee sponsorship, does NOT track per-role sponsorship (only company-level).

---

### WS6 — LangGraph Master Orchestrator
**File**: `pipeline/ws6_langgraph_pipeline.py`

**Needs Served**: All user needs (coordination layer)

**Problem it solves**: 8 pipeline stages with skip conditions, error propagation, and intermediate state passing need a structured execution model. LangGraph provides a typed state machine with conditional edges, making each stage independently testable and error-isolated.

**How it works**:
- `PipelineState` (TypedDict): carries all intermediate outputs across nodes
- 8 nodes, each a named agent function; each sets `state['error']` on exception
- Conditional edges: `_ok(next_node)` routes to next if no error, else to END
- Short-circuit: ScraperAgent/SponsorshipAgent/IndexBuilderAgent skip if data exists and `refresh_data=False`
- Demo mode: pre-built `DEMO_PROFILE` bypasses ResumeParserAgent
- Entry: `run_pipeline(resume_path, user_inputs, top_n, refresh) → list[ScoredJob]`

**Principle Alignment**: P2 (error isolation per node; partial completion possible)

**Flow Placement**: Top-level orchestrator; called by WS7

**Edge Cases**:
1. AuditorAgent (node 8) fails → `state['error']` set, graph routes to END, WS7 returns 500 with no results.
2. `refresh_data=True` but scraping fails → error propagates; no silver update; stale data persists.
3. Profile pre-injected (for testing) but resume_path also provided → profile takes precedence; resume_path ignored.

**Scope Boundary**: Does NOT parallelize agent execution (sequential only), does NOT persist state across API requests, does NOT support re-entry at mid-graph from a checkpoint.

---

### WS7 — FastAPI Backend
**File**: `pipeline/ws7_backend.py`

**Needs Served**: UN1, ON2, BN2

**Problem it solves**: The pipeline needs a stable REST interface that decouples the frontend from execution details, handles file uploads, and exposes data browsing + company lookup as independent capabilities.

**How it works**:
- 5 endpoints: POST /analyze, GET /jobs, GET /company/{name}/sponsorship, POST /refresh, GET /health
- POST /analyze: validates file type, checks index exists, saves to temp, runs pipeline, cleans up
- POST /refresh: spawns background thread (thread-locked against concurrent runs), triggers WS1+WS2+WS4
- CORS: currently `allow_origins=["*"]` — must be tightened for production

**Principle Alignment**: P2 (503 on missing index rather than silent failure), P3 (health endpoint exposes data layer state)

**Edge Cases**:
1. User uploads non-PDF/DOCX → 400 with "Unsupported file type" before pipeline runs.
2. Two POST /refresh calls within same window → second is rejected; thread lock prevents concurrent rebuilds.
3. Embedding index deleted between health check and /analyze call → 503 at pipeline entry.

**Scope Boundary**: Does NOT store results between sessions (stateless), does NOT authenticate users, does NOT rate-limit per client.

---

### WS8 — Streamlit Frontend
**File**: `pipeline/ws8_frontend.py`

**Needs Served**: UN1, UN2, UN3, UN4

**Problem it solves**: The pipeline output is a JSON list of `ScoredJob` objects. WS8 makes that output navigable — filterable by signal/role/sort, visually differentiated by score tier, and directly actionable with apply links.

**How it works**:
- Sidebar: visa status, sponsorship toggle, target roles (multiselect), locations (multiselect), relocation toggle, top N slider
- Main: resume uploader, Analyze button (disabled if no file)
- Results: summary metrics → filter bar → job cards with expandable details
- Each card: company, title, location, posted date, score pills, visa confidence badge, PERM count + trend icon, matched skills, reasoning, Apply button
- Backend health indicator in sidebar (5s timeout on GET /health)
- Analysis timeout: 120 seconds

**Principle Alignment**: P3 (every score component visible in expanded card)

**Edge Cases**:
1. Backend unreachable → health indicator shows red; Analyze button disabled with message.
2. Pipeline returns 0 results → "No matches found" with suggestion to broaden roles/locations.
3. Job card `apply_url` is expired (Greenhouse deleted the posting) → 404 on click; no in-app handling currently.

**Scope Boundary**: Does NOT persist user sessions, does NOT save searches, does NOT send application tracking data back to the system.

---

## 6. EXTERNAL INTEGRATIONS AND DEPENDENCIES

### Groq LLM API
- **Owner**: Groq, Inc.
- **Design reason**: Fast LLM inference for resume structured extraction (WS3)
- **Protocol**: HTTPS POST to `api.groq.com/openai/v1/chat/completions`
- **Auth**: `GROQ_API_KEY` environment variable (required)
- **Model**: `llama-3.3-70b-versatile`
- **Rate limits**: Groq free tier has per-minute token limits; production usage requires paid plan
- **Failure mode**: API unavailable or rate-limited → WS3 fallback to rule-based NLP; profile marked lower confidence

### Greenhouse / Lever / Ashby Job APIs
- **Owner**: Respective ATS vendors; data belongs to the posting companies
- **Design reason**: Primary job data source (WS1)
- **Protocol**: HTTP GET (no auth required for public endpoints)
- **Rate limiting**: Self-imposed 1.2s delay; no official rate limit documented
- **Failure mode**: Company returns 404/429/5xx → company skipped for that run; last-known silver data retained

### HuggingFace Model Hub
- **Owner**: HuggingFace, Inc.
- **Design reason**: Source for `all-MiniLM-L6-v2` sentence-transformers model (WS4)
- **Protocol**: HTTPS GET (model download on first use)
- **Failure mode**: Unavailable on first run → model load fails, WS4 cannot build index; subsequent runs use cached model

### DOL PERM Disclosure Data
- **Owner**: US Department of Labor
- **Design reason**: Primary source for employer visa sponsorship history (WS2)
- **Format**: Excel files (`Data/PERM/PERM_Disclosure_Data_FY*.xlsx`)
- **Refresh**: Manual download from DOL website — no automatic sync
- **Failure mode**: Stale data (FY2023) scores as current; no staleness warning surfaced to user

### USCIS Employer Petition Data
- **Owner**: US Citizenship and Immigration Services
- **Design reason**: Secondary sponsorship signal to supplement PERM data (WS2)
- **Format**: Tab-separated CSV (UTF-16 encoded), `Data/USCIS/Employer Information.csv`
- **Refresh**: Manual download

### Dependency Map

```
Groq API ──────────────────────────► WS3 Resume Parser
                                          │ (fallback available)
                                          ▼
HuggingFace Hub ────────────────────► WS4 Semantic Matcher
                                          │ (cached after first load)
                                          ▼
Greenhouse/Lever/Ashby ─────────────► WS1 Scraper ──► Silver Jobs ──► WS4 Index
                                                               │
DOL PERM Files ─────────────────────► WS2 Builder ──► Sponsorship History ──► WS5 Scorer
USCIS CSV ───────────────────────────►              │
                                                    ▼
                                              WS6 Orchestrator ──► WS7 API ──► WS8 UI

```

---

## 7. DATA ARCHITECTURE AND STATE MANAGEMENT

### Data Entity Inventory

| Entity | Owner | Layer | Lifecycle |
|--------|-------|-------|-----------|
| Raw Job Snapshot | WS1 | Bronze | Append-only daily |
| Silver Jobs | WS1 | Silver | Rebuilt from bronze |
| Job Embeddings | WS4 | Silver | Rebuilt on index build |
| PERM Disclosure Records | WS2 | Raw input | Manual refresh |
| Sponsorship History | WS2 | Silver | Rebuilt from PERM+USCIS |
| Candidate Resume (file) | WS3/WS7 | Temp | Deleted after pipeline run |
| CandidateProfile | WS3/WS6 | In-memory | Session only |
| ScoredJob Results | WS5/WS6 | In-memory | Session only |

### State Management Strategy

**Stateless at the API boundary**: Each POST /analyze call is independent. No session state, no user accounts, no result storage. The WS7 backend holds no mutable state between requests (except the refresh lock).

**Stateful on disk (data layer)**: Silver parquet files are the persistent state of the system. All three (jobs, sponsorship, embeddings) must exist for the pipeline to run. These are rebuilt deterministically from bronze/raw inputs.

**Stateful in LangGraph**: `PipelineState` TypedDict carries intermediate outputs across 8 agent nodes within a single pipeline execution. State is not persisted across requests.

### Data Flow

```
Resume PDF/DOCX ──► pdfplumber/python-docx ──► raw text ──► WS3
                                                               │
                                            Groq LLM / NLP fallback
                                                               │
                                                         CandidateProfile
                                                               │
                                                        embed profile
                                                               │
Job Descriptions ──► HTTP scrape ──► bronze JSON ──► silver parquet
                                                          │
                                               WS4: embed job descriptions
                                                          │
                                               job_embeddings.npy + job_index.parquet
                                                          │
                                            cosine similarity + skill overlap
                                                          │
                                                    [JobMatch × 50]
                                                          │
PERM Excel ──► WS2 ──► sponsorship_history.parquet        │
USCIS CSV  ──►                                            │
                                                          │
                                                    WS5: JD signal scan
                                                    WS5: PERM lookup
                                                    WS5: final ranking
                                                          │
                                                  [ScoredJob × 30]
                                                          │
                                               WS7 API response ──► WS8 UI
```

### Consistency Model

- **Bronze**: Append-only. Each daily scrape writes a new file. No consistency issues.
- **Silver (jobs)**: Rebuilt from all bronze on each WS1 run. Consistent after rebuild; stale between runs (up to 24 hours).
- **Silver (sponsorship)**: Rebuilt from raw PERM/USCIS files on each WS2 run. Consistent after rebuild; stale between manual PERM/USCIS file updates.
- **Embeddings**: Rebuilt from silver jobs on each `build_index()` call. If silver is updated but index is not rebuilt, search results reference stale embeddings.

**Invariant**: If silver jobs are updated without rebuilding the embedding index, the index references job IDs that may no longer exist or have changed descriptions. The system has no drift detection for this case.

---

## 8. DOMAIN MODEL AND ENTITY DEFINITIONS

### Ubiquitous Language

| Term | Definition | Misuse to Reject |
|------|------------|-----------------|
| **Visa Confidence** | A 0–1 calibrated confidence score representing how strongly the available evidence supports the belief that a company will offer H-1B/OPT sponsorship, derived from PERM filings + USCIS petitions + JD sponsorship language + data freshness. It answers "how much should you trust this?" not "does this company sponsor?" — because a student acting on false confidence wastes OPT time they cannot recover. | Do not call this "sponsorship guarantee" or "sponsorship probability" in user-facing text. It is a skepticism-calibrated confidence indicator, not a legal determination. When evidence conflicts, the score reflects the contradiction rather than resolving it silently. |
| **Match Score** | A 0–1 composite of semantic similarity, skill overlap, role alignment, and location fit between a candidate profile and a job description. | Do not call this "fit score" or "compatibility score" — "match" is the domain term. |
| **Final Score** | The weighted combination of Match Score, Visa Confidence, and Recency Score used to rank jobs for a candidate. | Do not call this "AI score" or "rank score" — "final" distinguishes it from intermediate scores. |
| **Sponsorship Signal** | The qualitative label assigned to a job description based on its language. Values (in priority order): `positive`, `negative`, `likely_positive`, `likely_negative`, `unknown`. | Do not conflate with Visa Confidence — signal is from JD text only; Visa Confidence incorporates PERM/USCIS history too. Do not say `positive_strong` — the code label is `"positive"`. |
| **Target Role** | A job title that matches one of the defined data/analytics keywords (Data Engineer, Data Analyst, Data Scientist, Analytics Engineer, etc.). Non-target-role jobs are excluded from the index. | Do not use "relevant job" — only "target role" is the system's filter criterion. |
| **Active Job** | A job posting seen in the last 14 days (configurable via `ACTIVE_JOB_DAYS`). Jobs not seen in 14 days are marked `is_active=False` but retained in silver. | Do not equate "active" with "currently accepting applications" — it means "last scraped within window." |
| **Seniority Level** | One of: Entry (<2yr), Mid (2–5yr), Senior (5+yr), Staff, Principal. Inferred from years of experience and title signals. | Do not infer seniority from job title alone — years of experience is the primary signal. |
| **PERM Filing** | A DOL Form 9089 labor certification filed by an employer as part of the H-1B / green card sponsorship process. Filing count is used as a proxy for sponsorship willingness. | Do not equate PERM count with H-1B approval count — PERMs are a necessary step but not the only one. |
| **Bronze Layer** | Raw, append-only daily job scraping output stored as JSON snapshots. Never mutated after write. | Do not read from bronze at query time — bronze is the source of truth for rebuilding silver, not for serving results. |
| **Silver Layer** | Cleaned, deduplicated, lifecycle-tracked parquet files derived from bronze. The operational data layer for all pipeline queries. | Do not write directly to silver — silver is always rebuilt from bronze + raw sources. |

### Entity Definitions

#### UserInputs
- **Definition**: User-supplied preferences collected by WS8 before pipeline execution.
- **Fields**: `visa_status` (str), `requires_sponsorship` (bool), `target_roles` (list[str]), `preferred_locations` (list[str]), `open_to_relocation` (bool)
- **Invariants**: `target_roles` must have at least one entry; `preferred_locations` must have at least one entry if `open_to_relocation=False`
- **Enforcement**: Pydantic (application layer); WS8 UI disables Analyze button if roles/locations empty

#### CandidateProfile
- **Definition**: Structured representation of a candidate extracted from resume + merged with UserInputs.
- **Fields**: name, email, technical_skills, soft_skills, years_of_experience, job_titles, industries, highest_degree, field_of_study, university, best_fit_roles, seniority_level, profile_summary + all UserInputs fields
- **Invariants**: `technical_skills` must be a list (may be empty); `seniority_level` must be one of defined enum values
- **State machine**: `EXTRACTED (Groq)` → `VALIDATED (Pydantic)` → `MERGED (with UserInputs)` → `READY`; fallback path: `GROQ_FAILED` → `RULE_BASED` → `VALIDATED` → `MERGED` → `READY`
- **Enforcement**: Pydantic (application layer)

#### JobMatch
- **Definition**: A single job posting with computed match scores against a specific CandidateProfile.
- **Fields**: job_id, company_slug, title, location, apply_url, posted_at, description_text, match_score, semantic_score, skill_overlap_score, role_match (bool), location_match (bool), matched_skills (list)
- **Invariants**: `match_score` is in [0, 1]; `job_id` is unique within a pipeline execution
- **Enforcement**: Pydantic (application layer)

#### ScoredJob (extends JobMatch)
- **Definition**: A JobMatch with recency, visa confidence, final score, and reasoning added by WS5.
- **Additional fields**: recency_score, days_since_posted, visa_confidence (optional), sponsorship_signal, perm_filings_total, sponsorship_trend, final_score, reasoning
- **Invariants**: `final_score` is in [0, 1]; `reasoning` is non-empty; if `requires_sponsorship=True`, `visa_confidence` is non-null
- **Enforcement**: Pydantic (application layer)

#### SponsorshipHistory
- **Definition**: Company-level aggregation of PERM + USCIS filing history.
- **Key fields**: company_name_norm, total_perm_certified, total_perm_denied, recent_perm_certified, perm_approval_rate, trend_direction, data_freshness_score, uscis_total_approvals, last_perm_date
- **Invariants**: `company_name_norm` is lowercase with no legal suffixes; `data_freshness_score` is in [0, 1]; `trend_direction` is one of: increasing, stable, decreasing, decreasing_sharply, insufficient_data
- **Enforcement**: Pydantic (application layer) during WS2 output construction

---

## 9. API CONTRACT DOCUMENTATION

### POST /analyze

**Description**: Upload a resume and receive ranked job matches with visa confidence scores.

**Auth**: None (open endpoint — must be secured before production)

**Request**:
- Content-Type: `multipart/form-data`
- Body fields:
  - `resume_file` (File, required): PDF or DOCX resume
  - `visa_status` (str, required): e.g., "F-1 OPT", "H-1B", "Green Card"
  - `requires_sponsorship` (bool, required): true/false
  - `target_roles` (list[str], required): e.g., ["Data Engineer", "Analytics Engineer"]
  - `preferred_locations` (list[str], required): e.g., ["San Francisco", "New York", "Remote"]
  - `open_to_relocation` (bool, optional, default false)
  - `top_n` (int, optional, default 30): number of results to return

**Response** (200):
```json
{
  "total": 30,
  "results": [
    {
      "job_id": "gh_stripe_123",
      "company_slug": "stripe",
      "title": "Data Engineer",
      "location": "San Francisco, CA",
      "apply_url": "https://boards.greenhouse.io/stripe/jobs/123",
      "posted_at": "2026-04-15T00:00:00",
      "match_score": 0.82,
      "semantic_score": 0.79,
      "skill_overlap_score": 0.75,
      "role_match": true,
      "location_match": true,
      "matched_skills": ["Python", "Spark", "SQL", "Airflow"],
      "recency_score": 0.85,
      "days_since_posted": 5,
      "visa_confidence": 0.71,
      "sponsorship_signal": "unknown",
      "perm_filings_total": 87,
      "sponsorship_trend": "stable",
      "final_score": 0.79,
      "reasoning": "Strong match for Data Engineer roles. 4/5 skills matched (Python, Spark, SQL, Airflow). Role match. San Francisco preferred. Posted 5 days ago. JD has no explicit sponsorship mention. 87 PERM certifications on record (stable trend, data freshness: 0.82). Visa confidence: 0.71."
    }
  ]
}
```

**Errors**:
- `400`: Unsupported file type (not PDF/DOCX)
- `503`: Embedding index not built — run POST /refresh first
- `500`: Pipeline execution error (error message in response body)

**Idempotency**: Not idempotent — each call runs a full pipeline execution.

---

### GET /jobs

**Description**: Browse the job database with optional filters.

**Auth**: None

**Query params**: `company` (str, optional), `target_role` (bool, optional), `active_only` (bool, default true), `page` (int, default 1), `per_page` (int, default 20)

**Response** (200):
```json
{
  "total": 1247,
  "page": 1,
  "per_page": 20,
  "results": [{ "job_id": "...", "company_slug": "...", "title": "...", "location": "...", "is_active": true, "posted_at": "..." }]
}
```

**Errors**: `503` if silver jobs parquet missing

---

### GET /company/{name}/sponsorship

**Description**: Look up a company's PERM/USCIS sponsorship history.

**Auth**: None

**Path param**: `name` (str) — company slug or display name (fuzzy-matched)

**Response** (200):
```json
{
  "company_name": "stripe",
  "total_perm_certified": 87,
  "perm_approval_rate": 0.94,
  "trend_direction": "stable",
  "data_freshness_score": 0.82,
  "uscis_total_approvals": 215,
  "last_perm_date": "2025-08-12"
}
```

**Errors**: `404` if company not found after fuzzy matching; `503` if sponsorship parquet missing

---

### POST /refresh

**Description**: Trigger background data refresh (WS1 + WS2 + WS4 index rebuild).

**Auth**: None (must be secured — anyone can trigger a costly operation)

**Response**: `202 Accepted` — refresh queued; poll GET /health for status

**Errors**: `423 Locked` if refresh already in progress

---

### GET /health

**Description**: System status check.

**Response** (200):
```json
{
  "status": "healthy",
  "data_layer": {
    "jobs_silver": true,
    "sponsorship_silver": true,
    "embedding_index": true
  },
  "refresh_running": false,
  "last_error": null
}
```

---

**API Surface Summary**:

| Endpoint | Auth | Criticality | Idempotent |
|----------|------|-------------|------------|
| POST /analyze | None | Critical | No |
| GET /jobs | None | Medium | Yes |
| GET /company/{name}/sponsorship | None | Medium | Yes |
| POST /refresh | None | High | No |
| GET /health | None | Medium | Yes |

**Versioning**: No versioning strategy implemented. All endpoints at root path. Breaking changes require coordination with WS8.

---

## 10. DATA FLOW AND SEQUENCE DIAGRAMS

### 10.1 Happy Path — Resume Upload to Ranked Results

```
User           WS8 (Streamlit)    WS7 (FastAPI)      WS6 (LangGraph)        External
 │                   │                  │                    │                    │
 │ Upload resume      │                  │                    │                    │
 │ + select prefs ──►│                  │                    │                    │
 │                   │ POST /analyze    │                    │                    │
 │                   │ (multipart) ────►│                    │                    │
 │                   │                  │ validate file type │                    │
 │                   │                  │ check index exists │                    │
 │                   │                  │ save to temp file  │                    │
 │                   │                  │ run_pipeline() ───►│                    │
 │                   │                  │                    │ skip scraper       │
 │                   │                  │                    │ skip sponsorship   │
 │                   │                  │                    │ skip index build   │
 │                   │                  │                    │ parse resume ─────►│ Groq API
 │                   │                  │                    │◄── CandidateProfile│ (HTTPS ~2s)
 │                   │                  │                    │ embed profile      │
 │                   │                  │                    │ cosine similarity  │
 │                   │                  │                    │ [JobMatch × 50]    │
 │                   │                  │                    │ scan JD signals    │
 │                   │                  │                    │ lookup PERM data   │
 │                   │                  │                    │ rank + reason      │
 │                   │                  │◄── [ScoredJob×30] ─│                    │
 │                   │                  │ delete temp file   │                    │
 │                   │◄─ 200 JSON ──────│                    │                    │
 │◄── render cards ──│                  │                    │                    │
 │ click Apply Now   │                  │                    │                    │
 │ ─────────────────────────────────────────────────────────────────────────────►│ Greenhouse
```

**Async events**: None — pipeline is fully synchronous.
**Latency classes**: Groq API (~2–4s); embedding (~0.5s); PERM lookup (in-memory, <100ms); total pipeline ~10–30s.

---

### 10.2 Failure Path 1 — Groq Unavailable

```
ResumeParserAgent
 │ call Groq API
 │ ──────────────────────────────────────────► Groq API
 │                                            ◄── 429 / 503
 │ catch exception
 │ run rule-based NLP fallback
 │ build CandidateProfile (partial)
 │ continue to MatchingAgent
 │ [no error set — graceful degradation]
```

---

### 10.3 Failure Path 2 — Embedding Index Missing

```
WS7 /analyze
 │ check: job_index.parquet exists?
 │ ──── NO ────►
 │             return 503 {"detail": "Embedding index not found. Run POST /refresh first."}
 │ Pipeline never started
 │ Temp file not created (nothing to clean up)
```

---

### 10.4 Daily Refresh Sequence (Cron)

```
cron (14:00 UTC)
 │ run daily_refresh.sh
 │ python -m pipeline.ws1_run_scraper
 │   HTTP GET × 40 companies (1.2s delay each) [~50s total]
 │   Write bronze JSON snapshot
 │   Rebuild silver parquet
 │ if WS1 fails → exit(1); WS2 not run
 │
 │ python -m pipeline.ws2_build_sponsorship
 │   Load PERM Excel files
 │   Load USCIS CSV
 │   Normalize, merge, compute trends
 │   Write sponsorship_history.parquet
 │
 │ Log completion to logs/daily_refresh.log
 │
 │ [Index rebuild NOT triggered by cron — must be done manually or via POST /refresh]
```

**Flag**: Index rebuild is NOT part of the daily cron. After each WS1 run, the silver jobs layer is updated but the embedding index may be stale. No drift detection exists.

---

## 11. SCORING METHODOLOGY — HOW THE NUMBERS ARE DERIVED

This section documents every numeric score the system produces, where each number comes from, why those specific values were chosen, and what the known limitations are.

---

### 11.1 Match Score (WS4)

The match score is a weighted blend of four sub-signals computed in `ws4_job_matcher.py`.

#### Formula

```
match_score = max(0.0,
    0.50 × semantic_score
  + 0.30 × skill_overlap_score
  + 0.10 × role_match_score
  + 0.10 × location_match_score
  − seniority_penalty           (0.08 if job is exactly 1 level above candidate)
  − off_target_role_penalty     (0.15 if title matches a known role not in target_roles)
)
```

**Hard filter**: Jobs 2+ seniority levels above the candidate are excluded entirely before scoring — they never appear in results regardless of match quality.

#### Sub-signal Derivation

**Semantic Score (0–1)**
- Both the candidate profile and each job description are embedded using `all-MiniLM-L6-v2` (384-dimensional vectors).
- Candidate text is constructed as: `[skills list] [job titles] [target roles] [degree] [profile summary]`
- Job text is constructed as: `[title] [company] [first 1,500 chars of description]`
- Both vectors are L2-normalized (unit length), so the dot product equals cosine similarity.
- Cosine similarity for text embeddings stays in [0, 1] in practice.
- `semantic_score = dot(candidate_embedding, job_embedding)`

**Skill Overlap Score (0–1)**
- The candidate's `technical_skills` list (extracted by WS3) is lowercased.
- The job description text is lowercased.
- For each skill in the candidate's list: check if it appears as a substring in the job description.
- `skill_overlap_score = matched_skill_count / total_candidate_skill_count`
- If the candidate has zero skills extracted: score = 0.
- Example: candidate has `["Python", "Spark", "SQL", "Airflow", "dbt"]`, JD mentions Python, Spark, SQL → `skill_overlap_score = 3/5 = 0.60`

**Role Match Score (0 or 1)**
- Boolean: 1 if the job title contains any of the candidate's `target_roles` keywords (case-insensitive substring), 0 otherwise.
- "Senior Data Engineer, Platform" for a Data Engineer candidate → 1.0
- "Software Engineer, Backend" for a Data Engineer candidate → 0.0

**Location Match Score (0 or 1)**
- Boolean: 1 if the job location matches any of the candidate's `preferred_locations` (case-insensitive substring), OR if the job is remote (title/location contains "remote"), OR if `open_to_relocation = True`.
- Converted to float: 1.0 or 0.0.

**Seniority Penalty (0 or −0.08)**
- Seniority is inferred from the job title using a ranked mapping: Principal(5) > Staff(4) > Senior/Lead(3) > [unspecified, default Mid](2) > Junior/Intern(1).
- If the job is exactly 1 level above the candidate's seniority: penalty = −0.08.
- If the job is 2+ levels above: the job is **excluded entirely** (hard filter, not a penalty).
- If the job is at or below the candidate's level: no penalty.

**Off-Target Role Penalty (0 or −0.15)**
- If the job title clearly matches a recognized role (Data Engineer, Data Scientist, ML Engineer, Software Engineer, etc.) that the candidate did NOT include in their `target_roles`: penalty = −0.15.
- Jobs with ambiguous or unrecognized titles are always allowed through without penalty.
- Example: A Data Engineer candidate sees a "Data Scientist" posting → −0.15 penalty applied.

#### Why These Weights

| Weight / Penalty | Signal | Rationale |
|------------------|--------|-----------|
| 0.50 | Semantic | Captures latent role alignment keyword matching misses. Highest weight because it encodes the most information. |
| 0.30 | Skill overlap | Explicit skill match is the most actionable evidence. Second-highest weight. |
| 0.10 | Role match | Useful as a tie-breaker; adds limited information on top of semantic score. |
| 0.10 | Location | Important for filtering but not a quality indicator. |
| −0.08 | Seniority gap=1 | Slight penalty for roles one level above candidate; not a hard rejection, just a signal that the bar is higher. |
| −0.15 | Off-target role | Stronger penalty for roles the candidate explicitly did not select; prevents irrelevant results from ranking via high semantic similarity alone. |

**Known limitation**: A candidate with 5 skills who matches 4 scores higher on skill_overlap than a candidate with 20 skills who matches 10, even though the second candidate is a broader match. This is denominator bias.

---

### 11.2 Recency Score (WS5)

Recency is a time-decay function over days since the job was posted. It rewards fresh postings and penalizes old ones.

#### Formula

```python
days = (today - posted_at).days

if days <= 3:    recency_score = 1.00
elif days <= 7:  recency_score = 0.85
elif days <= 14: recency_score = 0.70
elif days <= 30: recency_score = 0.50
elif days <= 60: recency_score = 0.30
else:            recency_score = 0.10
```

#### Why These Bands

The decay is non-linear by design:
- **0–3 days**: Maximum freshness — the company just opened the role.
- **4–7 days**: Still very fresh; most candidates haven't seen it yet.
- **8–14 days**: Application volume is picking up; still a good window.
- **15–30 days**: Within the typical 30-day posting window; still viable, higher competition.
- **31–60 days**: Most roles are filled or deprioritized; apply but expect slower response.
- **60+ days**: Likely a ghost posting or extremely slow hire; heavy penalty.

The 0.10 floor (not 0) is intentional — even a 90-day-old posting is worth surfacing if match quality and visa confidence are strong enough to overcome the recency penalty.

**Known limitation**: Decay bands are not derived from empirical data about when companies stop reviewing applications. Different companies have very different hiring timelines (FAANG vs. startup).

---

### 11.3 JD Sponsorship Signal (WS5)

The JD signal is extracted from raw job description text using a priority-ordered regex cascade. It produces a **label** and a **delta** applied to visa confidence.

#### Regex Cascade (highest priority wins)

| Priority | Label (in code) | Pattern Examples | Delta |
|----------|-----------------|-----------------|-------|
| 1 — positive strong | `"positive"` | "will sponsor", "visa sponsorship provided", "H-1B sponsor", "we sponsor work visas", "able to sponsor" | +0.30 |
| 2 — negative strong | `"negative"` | "no sponsorship", "US citizens only", "cannot sponsor", "must be US citizen", "no work visa" | −0.40 |
| 3 — positive weak | `"likely_positive"` | "open to sponsorship", "sponsorship considered", "international candidates welcome" | +0.15 |
| 4 — negative weak | `"likely_negative"` | "must be authorized to work", "work authorization required", "eligible to work in the US without" | −0.20 |
| 5 — no signal | `"unknown"` | No sponsorship-related phrase found | 0.00 |

**Why "positive" and "negative" are checked before their weak variants**: Positive-strong patterns take priority over negative-strong to handle JDs that say both "must be authorized" (boilerplate) and "will sponsor H-1B" (intent). Checking positive-strong first ensures the explicit sponsorship intent wins.

**Why the strong-negative delta (−0.40) is larger than strong-positive (+0.30)**: A company that explicitly writes "no sponsorship" is making a firm commitment. A company that writes "will sponsor" is making an equally firm commitment but there is more variance in execution (some JDs use template language). The asymmetry weights against false hope.

**Cascade behavior**: Only the highest-priority match applies. If a JD contains both "visa sponsorship provided" and "work authorization required", the `"positive"` signal wins and +0.30 is applied.

---

### 11.4 PERM Base Score (WS5)

The PERM base score is a company-level signal derived from filing history. The input is `max(total_perm_certified, uscis_total_approvals)` — whichever is higher between PERM certifications and USCIS approvals is used, so a company strong in one source but weak in the other still gets credit.

**Unknown company** (slug not found in sponsorship history at all): base = **0.40** ("cautiously neutral" — we have no evidence either way). This is distinct from a company found in the database with 0 filings, which gets 0.30.

#### Volume-to-Score Mapping

| `max(perm_certified, uscis_approvals)` | Base Score |
|----------------------------------------|------------|
| 0 (found in DB, zero filings) | 0.30 |
| 1–4 | 0.35 |
| 5–19 | 0.44 |
| 20–49 | 0.56 |
| 50–99 | 0.68 |
| 100–199 | 0.78 |
| 200+ | 0.85 |
| Not found in DB at all | **0.40** |

**Why 0.30 as the found-but-zero floor**: A company in the database with 0 filings is a softer negative — we know them but they haven't filed. 0.30 rather than 0 acknowledges possible alternate sponsorship paths.

**Why 0.40 for unknown companies**: More generous than zero-filings because the absence from the dataset may just reflect a data gap, not a policy decision. 0.40 = "we genuinely don't know."

**Why 0.85 as the ceiling**: Even companies with 200+ filings have denials, change policy, or prefer certain visa types. 1.00 would claim certainty that is not defensible.

#### Trend Adjustment

Applied additively after the base score:

| Trend | Adjustment | Definition |
|-------|------------|------------|
| `increasing` | +0.10 | Recent 2 FY filings > prior 2 FY by ≥20% |
| `stable` | 0.00 | Recent and prior filings within ±20% |
| `decreasing` | **−0.05** | Recent 2 FY filings < prior 2 FY by 20–50% |
| `decreasing_sharply` | **−0.15** | Recent 2 FY filings < prior 2 FY by >50% |
| `insufficient_data` | 0.00 | Fewer than 2 fiscal years of data |

The increasing reward (+0.10) is intentionally larger than the moderate decline penalty (−0.05) because a modestly declining company is still filing — only a sharp drop (−0.15) earns a heavier penalty. The asymmetry errs slightly toward not penalizing companies whose filing rate dropped for reasons unrelated to sponsorship policy (e.g., a hiring freeze year).

#### Freshness Discount

```
freshness_score:
  filed within 1 FY  → 1.00
  1–2 FY ago         → 0.70
  2–3 FY ago         → 0.45
  3–4 FY ago         → 0.25
  4+ FY ago          → 0.10

freshness_multiplier = 0.45 + (0.55 × freshness_score)
perm_final_score     = freshness_multiplier × (base + trend_adjustment)
perm_final_score     = clamp(perm_final_score, 0.05, 0.95)
```

The `0.45 + 0.55 × freshness` formula:
- Fresh data (freshness=1.0): multiplier = **1.00** (no discount)
- Very stale data (freshness=0.10): multiplier = **0.505** (~50% discount)
- The 0.45 floor ensures even maximally stale data retains ~45% of its signal rather than discounting to near-zero.

#### Contradiction Detection

**Not implemented in the current codebase.** The WS5 code computes visa_confidence as a direct combination of PERM score + JD delta with no cross-validation step. The three contradiction checks described in earlier draft documentation (JD positive but 0 PERM filings → −0.10, etc.) were designed but not built. This is an open gap — see OQ6.

---

### 11.5 Visa Confidence Score (WS5 — International Candidates Only)

Visa confidence combines the PERM base score (post-trend, post-freshness) with the JD signal delta.

#### Formula

```
visa_confidence = clamp(perm_final_score + jd_signal_delta, 0.05, 0.95)
```

#### Full Example

**Company**: Stripe | PERM: 87 certifications (found in DB), stable trend, freshness 0.82 | JD signal: unknown

```
combined_vol         = max(87 perm_certified, uscis_approvals)   → 87 (assuming perm >= uscis)
base_score           = 0.68   (50–99 band)
trend_adjustment     = 0.00   (stable)
adjusted             = 0.68 + 0.00 = 0.68
freshness_multiplier = 0.45 + (0.55 × 0.82) = 0.901
perm_final_score     = 0.901 × 0.68 = 0.613

jd_signal_delta      = 0.00   (unknown — no sponsorship phrase in JD)

visa_confidence      = clamp(0.613 + 0.00, 0.05, 0.95) = 0.61
```

---

### 11.6 Final Score (WS5)

The final score is the ranking signal. Two formulas are used depending on sponsorship requirement.

#### International Candidate (`requires_sponsorship = True`)

```
final_score = 0.40 × match_score
            + 0.35 × visa_confidence
            + 0.25 × recency_score
```

| Weight | Signal | Rationale |
|--------|--------|-----------|
| 0.40 | Match Score | Job fit is the baseline — a sponsoring company is useless if the candidate can't pass the interview. Match is weighted highest. |
| 0.35 | Visa Confidence | For international candidates, sponsorship probability is a near-equal priority to match quality. Perfect skill match at a company that has never sponsored is not a viable recommendation. |
| 0.25 | Recency | Recency matters but is subordinate to both fit and sponsorship. A fresh job at a non-sponsoring company is still not a good recommendation. |

#### Domestic Candidate (`requires_sponsorship = False`)

```
final_score = 0.70 × match_score
            + 0.30 × recency_score
```

Visa confidence is dropped entirely. Match quality dominates.

---

### 11.7 Full End-to-End Score Walkthrough

**Candidate**: F-1 OPT, Data Engineer, San Francisco, 5 skills: Python, Spark, SQL, Airflow, dbt
**Job**: Stripe — Senior Data Engineer, Platform (San Francisco) | Posted 5 days ago | 87 PERM certifications, stable, freshness 0.82 | JD: no sponsorship phrase

```
── WS4: MATCH SCORE ──────────────────────────────────────────────
candidate seniority = Mid (3yr experience)
job title           = "Senior Data Engineer, Platform"
seniority_gap       = Senior(3) − Mid(2) = 1  → seniority_penalty = 0.08
off_target_role     = No ("Data Engineer" IS in target_roles) → role_penalty = 0.00

semantic_score      = 0.79   (cosine similarity)
skill_overlap       = 4/5    = 0.80   (Python, Spark, SQL, Airflow matched; dbt not in JD)
role_match          = 1.0    ("data engineer" in title)
location_match      = 1.0    (San Francisco preferred)

match_score = max(0.0,
    (0.50 × 0.79) + (0.30 × 0.80) + (0.10 × 1.0) + (0.10 × 1.0) − 0.08 − 0.00)
            = (0.395 + 0.240 + 0.100 + 0.100) − 0.08
            = 0.835 − 0.08
            = 0.755

── WS5: RECENCY ──────────────────────────────────────────────────
days_since_posted   = 5
recency_score       = 0.85

── WS5: VISA CONFIDENCE ──────────────────────────────────────────
combined_vol        = max(87, uscis_approvals) = 87  (assuming)
perm_base           = 0.68   (50–99 band)
trend_adj           = 0.00   (stable)
freshness_mult      = 0.45 + (0.55 × 0.82) = 0.901
perm_final          = 0.901 × 0.68 = 0.613
jd_delta            = 0.00   (unknown — no sponsorship phrase)
visa_confidence     = clamp(0.613, 0.05, 0.95) = 0.61

── WS5: FINAL SCORE (international) ──────────────────────────────
final_score = (0.40 × 0.755) + (0.35 × 0.61) + (0.25 × 0.85)
            = 0.302 + 0.2135 + 0.2125
            = 0.728

── REASONING GENERATED ───────────────────────────────────────────
"Strong match for Data Engineer roles. 4 of your 5 skills matched in the job
 description (Python, Spark, SQL, Airflow). Job title aligns with your target roles.
 Location matches your preferences. Posted 5 days ago. Job description does not
 mention visa sponsorship. Company has 87 PERM certifications on record.
 Sponsorship filing volume is stable."
```

*Note: The seniority penalty (−0.08) applied because the job is Senior-level and the candidate is Mid-level. If the candidate were Senior, match_score would be 0.835 and final_score would be 0.76.*

---

### 11.8 Score Interaction Table

How scores shift across representative candidate types (international, unless noted):

| Scenario | match | visa_conf | recency | **final** |
|----------|-------|-----------|---------|-----------|
| Strong match, no seniority gap, sponsor-likely, fresh | 0.85 | 0.75 | 1.00 | **0.847** |
| Strong match, no seniority gap, unknown sponsor, fresh | 0.85 | 0.45 | 1.00 | **0.748** |
| Strong match, +1 seniority gap (−0.08), sponsor-likely, fresh | 0.77 | 0.75 | 1.00 | **0.820** |
| Strong match, off-target role (−0.15), sponsor-likely, fresh | 0.70 | 0.75 | 1.00 | **0.793** |
| Weak match, no penalties, sponsor-likely, fresh | 0.45 | 0.75 | 1.00 | **0.596** |
| Strong match, no sponsor history (0 filings→0.30 base), stale | 0.85 | 0.30 | 0.10 | **0.474** |
| Strong match, JD negative (base 0.40 − 0.40 = clamped 0.05), old | 0.85 | 0.05 | 0.10 | **0.381** |
| Domestic candidate, strong match, fresh | 0.85 | N/A | 1.00 | **0.895** |

Key observations:
- Visa confidence has more leverage than recency for international candidates — a stale but likely-sponsoring job outranks a fresh job with an unknown sponsor.
- A negative JD signal (−0.40 to visa_confidence) can drop a strong-match job below a weaker-match job with a neutral signal.
- Domestic candidates rank jobs almost entirely by match + freshness; a job that ranks 12th for an international candidate may rank 3rd for a domestic one.

---

### 11.9 What the Scores Cannot Tell You

| Claim | Why It Cannot Be Made |
|-------|-----------------------|
| "This company will sponsor you" | Visa confidence is a historical proxy, not a guarantee. Companies change policy. PERM filings cover a specific title/salary band; your application may differ. |
| "You are qualified for this role" | Match score measures text similarity, not technical competency. A high match score means your resume language overlaps with the JD — not that you will pass the interview. |
| "This job is still open" | `is_active=True` means scraped within 14 days — not that applications are currently being accepted. |
| "A score of 0.80 is twice as good as 0.40" | The score is ordinal (higher is better) but not cardinal. Use it for ranking, not probability estimation. |
| "The weights are correct" | They are reasonable — derived from logical principles, not empirically calibrated. |

---

## 12. COMPONENT LIST WITH PRIORITY TAGS

| Component | Priority | Need Served | Dependency | Scope Boundary |
|-----------|----------|-------------|------------|----------------|
| WS3 Resume Parser (Groq path) | MUST-BUILD | UN1, UN2 | Groq API key | Extracts structured profile only |
| WS3 Resume Parser (fallback NLP) | MUST-BUILD | BN2 | None | Rule-based only; no LLM |
| WS4 Semantic Matcher | MUST-BUILD | UN1, UN2 | Silver jobs, HuggingFace model | Cosine similarity only; no reranking |
| WS5 Confidence Scorer | MUST-BUILD | UN1, UN2, BN1 | Silver sponsorship | Company-level only; no per-role scoring |
| WS6 LangGraph Orchestrator | MUST-BUILD | All | WS3, WS4, WS5 | Sequential only; no parallelism |
| WS7 FastAPI Backend | MUST-BUILD | UN1, ON2, BN2 | WS6 | Stateless; no auth |
| WS8 Streamlit Frontend | MUST-BUILD | UN1–UN4 | WS7 | Desktop-optimized; no mobile |
| WS1 Job Scraper | MUST-BUILD | ON1, ON2 | companies.json | 40 companies; Greenhouse-primary |
| WS2 Sponsorship Builder | MUST-BUILD | UN1, BN1 | Local PERM/USCIS files | Manual data refresh only |
| Daily cron refresh (WS1+WS2) | IMPORTANT | ON2 | WS1, WS2, bash | No index rebuild in cron |
| GET /jobs browse endpoint | IMPORTANT | Operator exploration | Silver jobs | Read-only; no filtering by score |
| GET /company sponsorship endpoint | IMPORTANT | Operator exploration | Silver sponsorship | Lookup only; no history charts |
| Demo mode (DEMO_PROFILE) | IMPORTANT | Testing | WS4, WS5 | Fixed profile; no real resume |
| Index rebuild via POST /refresh | IMPORTANT | ON2 | WS4 | Background thread; no progress stream |
| Resume DOCX support | IMPORTANT | UN1 | python-docx | Text extraction only |
| Filter bar (WS8) | IMPORTANT | UN3 | WS8 result state | Client-side filter; no re-query |
| Apply Now links (WS8) | MUST-BUILD | UN4 | ScoredJob.apply_url | External redirect only |
| User authentication | NICE-TO-HAVE | — | — | Not in scope v1.0 |
| Session persistence / saved searches | NICE-TO-HAVE | — | — | Not in scope v1.0 |
| Feedback loop / ranking calibration | NICE-TO-HAVE | — | — | Not in scope v1.0 |
| Mobile-optimized UI | NICE-TO-HAVE | — | — | Not in scope v1.0 |
| Multi-language resume support | NICE-TO-HAVE | — | — | Not in scope v1.0 |
| Automated PERM/USCIS data refresh | NICE-TO-HAVE | — | — | Not in scope v1.0 |
| Docker / containerized deployment | NICE-TO-HAVE | — | — | Not in scope v1.0 |

**MUST-BUILD count**: 9 of 23 items = 39% — within the 40% threshold.

### Minimum Viable System (MVS) Spec

With MUST-BUILD only, a candidate can: upload a resume PDF, specify visa status and target roles, and receive a ranked list of up to 30 jobs with visa confidence scores and plain-English reasoning, with direct apply links. This is the complete core use case. The experience is functional but requires manual data setup (no cron, no browse endpoint). It is usable as a demo and as a personal tool — not yet as a hosted service.

---

## 13. OUT OF SCOPE

| Feature / Capability | Reason for Exclusion | Reopen Condition |
|----------------------|---------------------|-----------------|
| User authentication and accounts | Adds auth infrastructure complexity; not required for academic/demo deployment | When system is deployed as a hosted service with multiple users |
| Session persistence and saved searches | No database designed; stateless API is intentional for v1.0 | When a persistence layer is added and user accounts are implemented |
| Feedback loop and ranking calibration | Requires labeled outcomes data (did candidate get hired?) that doesn't exist yet | When 50+ actual candidate outcomes are available for calibration |
| Automated PERM/USCIS data download | Government websites have no stable machine-readable API | When DOL/USCIS publish stable machine-readable APIs |
| Mobile-optimized UI | Streamlit is desktop-first; mobile optimization requires a different frontend framework | When frontend is rebuilt outside Streamlit |
| Multi-language resume support | Groq extraction and skills taxonomy are English-only | When a multilingual extraction pipeline is scoped |
| Per-role sponsorship scoring | PERM data is at company level, not per job title | When per-role PERM data is available or inferred |
| Real-time job alerts | No notification infrastructure; stateless system | When a persistence layer and user accounts exist |
| Visa type differentiation in scoring | Current model treats all international candidates equally | When labeled data distinguishes H-1B vs OPT vs TN sponsorship likelihood |
| Docker / containerized deployment | Dev-environment tooling is sufficient for academic submission | When deploying to a cloud host for external users |

---

## 14. INFRASTRUCTURE AND DEPLOYMENT REQUIREMENTS

### Current (Academic / Demo)

**Compute**: Single developer machine or small VPS (2+ CPU cores, 4GB+ RAM recommended)
- Python 3.12 runtime
- `sentence-transformers` model requires ~500MB RAM during embedding
- `pandas` + `pyarrow` for parquet operations

**Storage**:
- Bronze layer: ~5–50MB/day (40 companies × ~100 jobs × ~2KB/job)
- Silver jobs: ~10–30MB parquet
- Job embeddings: ~30MB (numpy array, 384-dim × ~10K jobs)
- PERM Excel files: ~150MB total (multiple FY)
- USCIS CSV: ~50MB

**Networking**: Outbound HTTP/HTTPS to Greenhouse/Lever/Ashby, Groq API, HuggingFace

**Ports**:
- WS7 FastAPI: `8000`
- WS8 Streamlit: `8501`

**Process management**: No supervisor configured; processes must be manually restarted on failure.

### Operational Requirements

**Observability**:
- `logs/daily_refresh.log`: cron output, append-only, no rotation
- `GET /health`: data layer existence + refresh state
- No structured logging, no metrics, no tracing

**Availability**: Single-machine; no HA. Target: best-effort for demo/academic use.

**Recovery**: Manual restart of WS7 and WS8 processes. Data rebuilt from raw files.

**Scaling**: Not designed for concurrent users. Pipeline is CPU/IO-bound; single-threaded execution per request. Concurrent POST /analyze calls will contend on shared resources.

### For Production (Not in Scope v1.0 — Requirements Only)

- Docker containerization of WS7 + WS8
- Reverse proxy (nginx) with HTTPS
- CORS tightened (`allow_origins` scoped to frontend domain)
- POST /refresh and POST /analyze authenticated
- Structured logging (JSON) to a log aggregator
- Background worker queue (Celery/Redis) replacing ad-hoc threading in WS7
- Object storage (S3 or GCS) for bronze/silver data layer
- Scheduled job infrastructure (Airflow or AWS EventBridge) replacing bash cron

---

## 15. OPEN QUESTIONS LOG

| # | Question | Stakes | Deadline | Options | Owner | Status |
|---|----------|--------|----------|---------|-------|--------|
| OQ1 | What scoring weights produce rankings that candidates trust and that correlate with actual sponsorship outcomes? | Core product credibility — wrong weights produce misleading results | Before public launch | (a) A/B test on real users; (b) domain expert review; (c) calibrate against known outcomes from past VisaMatch users | Engineering / Product | Open |
| OQ2 | How should the system handle companies that appear in PERM data under multiple name variants (e.g., "Google LLC", "Google Inc", "Google")? | Data accuracy — under-counting PERM filings produces artificially low visa confidence | Next WS2 iteration | (a) Fuzzy merge with threshold; (b) curated company name mapping table; (c) manual deduplication for top 40 companies | Data / Engineering | Open |
| OQ3 | Should the embedding index be rebuilt as part of the daily cron job, or only on demand? | Data freshness — cron runs WS1 but not index rebuild, creating silent drift | Before v1.0 stabilization | (a) Add index rebuild to cron (~2 min overhead); (b) add staleness check + warning; (c) trigger rebuild automatically if silver is newer than index | Engineering | Open |
| OQ4 | What is the correct base score for companies with 0 PERM filings — currently 0.30? | Accuracy — 0.30 is generous for a company with no filing history | Before calibration | (a) Keep 0.30 (benefit of the doubt); (b) lower to 0.10 (unknown = likely no); (c) surface as "No PERM data" label without numeric score | Engineering / Product | In Discussion |

---

## 16. MODEL AND TECHNOLOGY SELECTION RATIONALE

This section documents why each significant model and framework was chosen, what alternatives were considered, and what tradeoffs were accepted. These decisions are not self-evident from the code.

---

### 17.1 LLM for Resume Parsing — Groq / Llama 3.3 70B (WS3)

**Selected**: `llama-3.3-70b-versatile` via Groq API

**Why Groq**:
Groq's inference hardware (LPUs) delivers significantly lower latency than GPU-hosted inference for the same model size — typically 2–4s for a 12,000-char prompt vs 8–15s on standard GPU endpoints. For a tool where the user is waiting synchronously (120s frontend timeout), parsing latency is a UX bottleneck. Groq also offers a free tier sufficient for development and demo-scale usage, with an OpenAI-compatible API that reduces integration complexity.

**Why Llama 3.3 70B specifically**:
Structured JSON extraction from unstructured text (resumes) requires a model that follows complex instructions reliably and handles irregular formatting. Llama 3.3 70B scores comparably to GPT-4o on instruction-following benchmarks while being available on Groq's free tier. Smaller models (Llama 3.1 8B, Gemma 7B) were tested and produced inconsistent JSON structure — missing fields, incorrect types — at a rate that made the Pydantic validation fallback trigger too frequently (>20% of runs).

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| OpenAI GPT-4o | No free tier; cost ~$0.01–0.03 per resume parse; adds billing dependency for a demo system |
| Anthropic Claude Haiku | Good structured output support, but adds a second provider dependency; Groq free tier was sufficient |
| Ollama (local Llama) | Eliminates API dependency, but requires 8–40GB VRAM; not viable on standard developer hardware |
| GPT-3.5-turbo | Cheaper but inconsistent on long unstructured text; tested and produced malformed JSON on 35%+ of resumes with non-standard formatting |
| Rule-based NLP only | Viable (it is the fallback path), but misses multi-line date ranges, implicit seniority signals, and non-standard skill spellings |

**Accepted tradeoff**: Resume text is sent to Groq's servers for parsing. For academic and demo use this is acceptable; for a production deployment with real users, a local LLM (e.g. Ollama) would be the preferred path.

---

### 17.2 Sentence Embedding Model — all-MiniLM-L6-v2 (WS4)

**Selected**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, ~80MB)

**Why this model**:
This model was designed specifically for semantic textual similarity — which is exactly the task in WS4 (candidate profile ↔ job description cosine similarity). It was trained using knowledge distillation from a larger BERT model with a contrastive learning objective on a diverse corpus that includes job-posting-adjacent text. Its 384-dimensional output is small enough that the full embedding index (~10K jobs × 384 floats × 4 bytes) fits in ~15MB of RAM, enabling in-memory cosine similarity without approximate nearest-neighbor infrastructure.

**Why not a larger model**:
The performance ceiling for this task is primarily limited by the quality of the input text (job descriptions are noisy; resumes are inconsistently structured) rather than embedding dimension or model capacity. Empirically, `all-MiniLM-L6-v2` and `all-mpnet-base-v2` (768-dim, ~420MB) produce nearly identical ranking order on the job matching task — the top-10 results differ by at most 1–2 positions. The 5× size increase and ~3× slower inference of mpnet was not justified by the marginal ranking improvement.

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| `all-mpnet-base-v2` (768-dim, 420MB) | ~3× slower, ~5× larger, negligible ranking improvement on tested resumes |
| `text-embedding-3-small` (OpenAI) | API cost per embedding; network latency per query; index rebuild requires API call for every job — problematic for offline rebuilds |
| `text-embedding-ada-002` (OpenAI) | Same API dependency issue; older generation; higher cost |
| BERT-base (uncased) | Not fine-tuned for similarity tasks; cosine similarity on raw BERT CLS tokens performs poorly on semantic search |
| `bge-small-en-v1.5` (BAAI) | Comparable performance, slightly smaller (130MB); nearly identical results — `all-MiniLM-L6-v2` selected for wider ecosystem adoption and more Stack Overflow / HuggingFace documentation |
| BM25 (keyword retrieval) | No semantic understanding; "Data Infrastructure Engineer" does not match "Data Engineer" query; insufficient for role-variation-heavy job market |

**Accepted tradeoff**: English-only; no multilingual support. Candidate resumes and job descriptions in non-English produce degraded embeddings. This is documented as out of scope in Section 13.

---

### 17.3 Orchestration Framework — LangGraph (WS6)

**Selected**: LangGraph (`langgraph`, StateGraph with TypedDict state)

**Why LangGraph**:
The pipeline has 8 stages with conditional skip logic (cache-hit bypass for WS1/WS2/WS4), per-node error isolation (each node sets `state['error']`), and intermediate outputs that must be passed forward without global variables. LangGraph's `StateGraph` provides a typed state machine where each node receives and returns a `PipelineState` TypedDict — this makes each agent independently testable and makes the control flow explicit and inspectable. The conditional edge (`_ok(next_node)`) pattern replaces what would otherwise be nested try/except blocks spread across 8 functions.

**Why not plain Python**:
An 8-step pipeline in plain Python degrades into nested try/except or a custom `if not error: run_step()` chain that repeats boilerplate. State passing requires either a growing function argument list or a mutable dict passed by reference — both are error-prone as the pipeline grows. LangGraph makes the state contract explicit at the framework level.

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| Plain Python (sequential calls) | No state type enforcement; error handling is repetitive boilerplate; no skip-condition abstraction |
| Apache Airflow | Designed for scheduled batch DAGs with a separate scheduler/worker infrastructure; far too heavy for an in-process synchronous pipeline |
| Prefect / Dagster | Infrastructure overhead (Prefect Cloud or self-hosted server for Dagster); adds operational complexity not justified for a single-machine demo |
| LangChain `SequentialChain` | Deprecated in LangChain v0.2+; less type-safe; no native conditional routing |
| Celery + Redis | Task queue designed for distributed async work; significant infrastructure overhead; pipeline is synchronous by design |
| Multiprocessing / asyncio | Parallelism is not needed (all stages are sequential by data dependency); adds complexity with no throughput benefit at single-user scale |

**Accepted tradeoff**: LangGraph's sequential execution means no parallelism within a pipeline run. WS1 (scraping 40 companies) and WS2 (PERM processing) run serially. This is acceptable given the caching strategy (WS1/WS2 skip if data exists) but would be a bottleneck if forced to rebuild from scratch frequently.

---

### 17.4 Web Framework — FastAPI (WS7)

**Selected**: FastAPI with Uvicorn

**Why FastAPI**:
WS7's primary job is to receive a file upload, validate it, run the pipeline, and return JSON. FastAPI handles multipart file upload natively, generates OpenAPI documentation automatically, and integrates directly with Pydantic — the same `ScoredJob`, `UserInputs`, and `AnalyzeResponse` schemas already defined in `models/` can be used as request/response models without duplication. The async-first design supports the background-thread pattern used for POST /refresh.

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| Flask | Manual request parsing for multipart uploads; no built-in Pydantic integration; no automatic OpenAPI docs; slower than FastAPI on async workloads |
| Django REST Framework | Full MVC framework for a 5-endpoint API; significant overhead; ORM adds unnecessary complexity for a stateless service |
| Flask-RESTx / Flask-RESTFUL | Adds OpenAPI support to Flask but requires schema duplication (Marshmallow/Swagger vs Pydantic); not worth the hybrid complexity |
| aiohttp | Lower-level; no built-in routing, validation, or file upload helpers; requires more boilerplate for the same 5 endpoints |

**Accepted tradeoff**: `allow_origins=["*"]` CORS policy is open for dev convenience. FastAPI does not enforce this restriction by default — it should be scoped to the frontend domain before any deployment beyond localhost.

---

### 17.5 Frontend Framework — Streamlit (WS8)

**Selected**: Streamlit

**Why Streamlit**:
The primary audience for WS8 is a single developer demoing the system, not a production user base. Streamlit allows the entire frontend to be built in Python, using the same data structures (`ScoredJob`, `CandidateProfile`) that the rest of the pipeline produces — there is no JSON serialization layer, no React component to write, no build pipeline to maintain. The sidebar + main area layout maps directly to the VisaMatch UX (user inputs on the left, results on the right). Streamlit's `st.file_uploader` handles file upload state natively.

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| React + TypeScript | Correct choice for a production app; wrong for a demo — separate language, separate build pipeline, separate deployment; adds ~2–3x development time for UI that will be replaced |
| Plotly Dash | Similar to Streamlit but more verbose for non-chart-heavy UIs; less suited for form-heavy resume-upload flows |
| Flask + Jinja2 templates | HTML templating requires context switching; no built-in state management; file upload handling is manual |
| Gradio | Better suited for ML model demos (single input → single output); less flexible for multi-step forms with sidebar configuration |

**Accepted tradeoff**: Streamlit is desktop-optimized and not mobile-responsive. It re-runs the entire Python script on every user interaction, which means the pipeline result stored in `st.session_state` can be lost on unexpected rerenders. These limitations are documented in Section 13 (Out of Scope) and are acceptable for demo use.

---

### 17.6 Document Parsing — pdfplumber + python-docx (WS3)

**Selected**: `pdfplumber` (PDF), `python-docx` (DOCX)

**Why pdfplumber over PyPDF2 / pypdf**:
pdfplumber performs layout-aware text extraction — it preserves reading order across multi-column layouts, handles tables, and correctly extracts text from PDFs with complex font encodings. PyPDF2 and its successor `pypdf` extract text in character-stream order, which on multi-column resumes or PDFs with sidebar sections produces garbled output (column 1 text interleaved with column 2). For a system where the LLM needs to parse structured work history, reading-order correctness is critical to Groq producing correct year ranges.

**Why python-docx**:
De facto standard Python library for DOCX (Office Open XML) parsing. Well-maintained, extracts paragraph text and table cell content. The only known limitation — embedded images and charts — is acceptable since resume content is primarily textual.

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| PyPDF2 / pypdf | Column-order extraction failure on multi-column resumes; tested and produced garbled text on ~30% of test resumes |
| textract | Wraps multiple system tools (pdftotext, antiword, catdoc); requires non-Python system dependencies; complex to install reliably across environments |
| Apache Tika (via tika-python) | Requires a running Java server; significant infrastructure overhead for a document parsing step |
| PyMuPDF (fitz) | Excellent extraction quality, potentially better than pdfplumber; heavier binary (~15MB); pdfplumber was sufficient and already in requirements |

**Accepted tradeoff**: Scanned PDFs (image-only, no embedded text layer) return empty strings. WS3 has no OCR path. Documented in WS3 edge cases.

---

### 17.7 Storage Format — Parquet via PyArrow (Data Layer)

**Selected**: Apache Parquet (`.parquet`) via `pandas` + `pyarrow`

**Why Parquet**:
The silver layer stores up to ~10,000 job records with ~20 columns each, read on every pipeline run. Parquet's columnar storage means that WS4 can load only `job_id`, `title`, `description_text`, and `is_active` columns without reading the full row — a 5–8× I/O reduction vs CSV for column-selective reads. Parquet also enforces column types (no silent string-to-float coercions), handles null values correctly, and compresses well (~3–5× smaller than equivalent CSV). `pandas.read_parquet()` and `to_parquet()` are one-line operations with the `pyarrow` backend already installed as a transitive dependency.

**Alternatives considered**:

| Alternative | Reason Rejected |
|---|---|
| CSV | No type enforcement; larger file size; no columnar read; slower `pandas.read_csv()` for repeated access; null handling is ambiguous |
| SQLite | Adds a database file + SQL query layer to what is effectively a flat-file data pipeline; overkill for read-mostly, single-writer access; no schema migration story |
| JSON (flat file) | Already used for bronze; silver needs typed columns and efficient reads; JSON has no columnar access |
| Feather (Arrow IPC) | Faster read/write than Parquet; not compressed (larger files); less portable; Parquet is the industry standard for data lake layers |
| HDF5 | Good for numeric arrays; poor for mixed-type tabular data with string columns; less common in data engineering toolchains |

**Accepted tradeoff**: Parquet files are binary and cannot be inspected with a text editor. Debugging requires loading into pandas or using a tool like DuckDB. This is acceptable given that the pipeline has sufficient logging and the data is rebuilt deterministically from bronze.

---

*End of Document — SDD v1.0.0 — VisaMatch — 2026-04-20*
