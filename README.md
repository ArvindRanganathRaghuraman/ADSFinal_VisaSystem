# ADSFinal_VisaSystem

AI-powered job matching for international students — ranks job postings by semantic fit, posting recency, and H-1B/PERM sponsorship confidence.

## Documentation

- [Technical Decisions](docs/technical_decisions.md) — why we chose Groq + Llama 3.3 70B, how the Groq API call works, fine-tuning considerations, and how job matching works end to end.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY

# Start the API backend
python -m uvicorn pipeline.ws7_backend:app --reload --port 8000

# Start the web UI (separate terminal)
streamlit run pipeline/ws8_frontend.py

# Or run the CLI directly
python run_pipeline.py --resume Resumes/your_resume.pdf --visa "F-1 OPT"
```

## Scraping Fresh Jobs

The job data does not auto-refresh. Run this after each scrape interval:

```bash
python -m pipeline.ws1_run_scraper                   # fetch latest jobs
python -m pipeline.ws4_job_matcher --build --force   # rebuild embeddings index
```

Then restart the backend so it picks up the new index.

## Pipeline Overview

| Stage | File | What it does |
|---|---|---|
| WS1 | `pipeline/ws1_run_scraper.py` | Scrape jobs from Greenhouse / Lever / Ashby |
| WS2 | `pipeline/ws2_build_sponsorship.py` | Process DOL PERM + USCIS data |
| WS3 | `pipeline/ws3_resume_parser.py` | Parse resume → structured profile via Groq |
| WS4 | `pipeline/ws4_job_matcher.py` | Semantic matching with sentence-transformers |
| WS5 | `pipeline/ws5_confidence_scorer.py` | Final ranking with visa + recency signals |
| WS6 | `pipeline/ws6_langgraph_pipeline.py` | LangGraph orchestration of WS3→WS4→WS5 |
| WS7 | `pipeline/ws7_backend.py` | FastAPI REST API |
| WS8 | `pipeline/ws8_frontend.py` | Streamlit web UI |

## Contributors

- [Akash Arokianathan](https://github.com/Akash-Arokianathan504)
