# VisaMatch — AI-Powered Job Matching for International Students

VisaMatch helps F-1 OPT / H-1B candidates find jobs at companies that actually sponsor visas. Upload your resume, set your preferences, and get a ranked list of matching jobs with a **visa sponsorship confidence score** for every result — backed by real DOL PERM and USCIS data.

---

## How It Works

The system runs a sequential 8-stage pipeline:

| Stage | Module | What it does |
|-------|--------|--------------|
| WS1 | `ws1_run_scraper.py` | Scrapes job listings from Greenhouse job boards |
| WS2 | `ws2_build_sponsorship.py` | Builds sponsorship history from DOL PERM + USCIS data |
| WS3 | `ws3_resume_parser.py` | Parses resume (PDF/DOCX) using Groq Llama 3.3 70B |
| WS4 | `ws4_job_matcher.py` | Semantic job matching via `all-MiniLM-L6-v2` embeddings |
| WS5 | `ws5_confidence_scorer.py` | Computes visa sponsorship confidence score |
| WS6 | `ws6_langgraph_pipeline.py` | Orchestrates WS3–WS5 via LangGraph StateGraph |
| WS7 | `ws7_backend.py` | FastAPI REST backend |
| WS8 | `ws8_frontend.py` | Streamlit candidate-facing UI |

---

## Prerequisites

- Python 3.12+
- Docker + Docker Compose
- A [Groq API key](https://console.groq.com) (free tier works)

---

## Run Locally

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd ADSFinal_VisaSystem
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create `.env` file

```bash
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### 3. Build the data layer (first time only)

```bash
python -m pipeline.ws2_build_sponsorship
python -c "from pipeline.ws4_job_matcher import build_index; build_index(force=True)"
```

### 4. Start the backend

```bash
uvicorn pipeline.ws7_backend:app --reload --port 8000
```

### 5. Start the frontend (new terminal)

```bash
streamlit run pipeline/ws8_frontend.py
```

Frontend: http://localhost:8501  
Backend docs: http://localhost:8000/docs

---

## Run with Docker Compose

```bash
# Copy and fill in your Groq key
cp .env.example .env

# Start both containers
docker compose up --build
```

Frontend: http://localhost:8501  
Backend: http://localhost:8000

To refresh job data after startup:
```bash
curl -X POST http://localhost:8000/refresh
```

---

## Pull from Docker (pre-built images)

Images are hosted on Google Artifact Registry:

```bash
# Authenticate
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull images
docker pull us-central1-docker.pkg.dev/project-18918c13-0693-4d59-90e/adsproject/backend:latest
docker pull us-central1-docker.pkg.dev/project-18918c13-0693-4d59-90e/adsproject/frontend:latest

# Run backend
docker run -p 8000:8000 \
  -e GROQ_API_KEY=gsk_your_key_here \
  us-central1-docker.pkg.dev/project-18918c13-0693-4d59-90e/adsproject/backend:latest

# Run frontend (new terminal)
docker run -p 8501:8501 \
  -e BACKEND_URL=http://localhost:8000 \
  us-central1-docker.pkg.dev/project-18918c13-0693-4d59-90e/adsproject/frontend:latest
```

---

## Live Demo

- **Frontend:** https://backendads-134643354783.us-east1.run.app *(Streamlit Cloud)*
- **Backend API:** https://backendads-134643354783.us-east1.run.app *(Google Cloud Run)*
- **API Docs:** https://backendads-134643354783.us-east1.run.app/docs

---

## Project Structure

```
ADSFinal_VisaSystem/
├── pipeline/
│   ├── ws1_run_scraper.py       # Job scraper
│   ├── ws2_build_sponsorship.py # Sponsorship index builder
│   ├── ws3_resume_parser.py     # LLM resume parser
│   ├── ws4_job_matcher.py       # Semantic matcher
│   ├── ws5_confidence_scorer.py # Visa confidence scorer
│   ├── ws6_langgraph_pipeline.py# LangGraph orchestrator
│   ├── ws7_backend.py           # FastAPI backend
│   └── ws8_frontend.py          # Streamlit frontend
├── Data/
│   ├── PERM/                    # DOL PERM disclosure files
│   ├── USCIS/                   # USCIS I-129 approval data
│   └── silver/                  # Pre-built sponsorship parquet
├── Scrapers/                    # Job scraping utilities
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload resume + preferences → ranked job list |
| `GET` | `/jobs` | Browse all scraped jobs |
| `GET` | `/company/{name}/sponsorship` | PERM + USCIS history for a company |
| `POST` | `/refresh` | Re-scrape jobs and rebuild index |
| `GET` | `/health` | Data layer status check |

---

## Tech Stack

- **LLM:** Groq Llama 3.3 70B (resume parsing)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Orchestration:** LangGraph StateGraph
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Data:** DOL PERM Disclosure Data + USCIS I-129 Records
- **Deployment:** Google Cloud Run + Streamlit Cloud
