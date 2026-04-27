from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.programming.flowchart import Document, MultipleDocuments
from diagrams.onprem.client import User, Client
from diagrams.onprem.network import Internet
from diagrams.generic.storage import Storage
from diagrams.aws.ml import Sagemaker

graph_attr = {
    "fontsize": "22",
    "bgcolor": "#f4f6fb",
    "fontcolor": "#1a1d2e",
    "pad": "1.5",
    "splines": "ortho",          # right-angle arrows → straight, aligned markers
    "nodesep": "0.9",
    "ranksep": "1.8",
    "rankdir": "LR",
    "concentrate": "false",
    "newrank": "true",
    "compound": "true",
}

node_attr = {
    "fontsize": "10",
    "fontcolor": "#1a1d2e",
    "style": "filled",
    "fillcolor": "#ffffff",
}

edge_attr = {
    "color": "#7b8cde",
    "fontsize": "9",
    "fontcolor": "#4a5568",
    "arrowsize": "0.8",
}

with Diagram(
    "VisaMatch – System Architecture",
    filename="Diagrams/architecture_diagram",
    outformat="png",
    show=True,
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    direction="LR",
):

    # ── User ──────────────────────────────────────────────────────────────────
    user = User("Candidate\n(Resume PDF/DOCX)")

    # ── Application Layer ─────────────────────────────────────────────────────
    with Cluster("Application Layer",
                 graph_attr={"bgcolor": "#e8f5e9", "fontcolor": "#1b5e20",
                             "rankdir": "TB"}):
        frontend = Client("WS8 · Streamlit Frontend\nJob cards · filters · uploader")
        backend  = FastAPI("WS7 · FastAPI Backend\nPOST /analyze")
        cli      = Python("run_pipeline.py\nCLI runner (WS3→WS4→WS5)")
        frontend >> Edge(color="#2e7d32", label="POST /analyze") >> backend
        cli      >> Edge(color="#2e7d32", style="dashed",
                        label="batch mode")                      >> backend

    # ── Data Sources ──────────────────────────────────────────────────────────
    with Cluster("Data Sources",
                 graph_attr={"bgcolor": "#e0f7fa", "fontcolor": "#006064",
                             "rankdir": "TB"}):
        ats   = Internet("Greenhouse / Lever\n/ Ashby ATS APIs")
        perm  = MultipleDocuments("DOL PERM\nDisclosures\n(FY2020–FY2025)")
        uscis = Document("USCIS H-1B\nAggregates CSV")

    # ── External AI Services ──────────────────────────────────────────────────
    with Cluster("External AI Services",
                 graph_attr={"bgcolor": "#fdecea", "fontcolor": "#b71c1c",
                             "rankdir": "TB"}):
        groq  = Internet("Groq API\n(Llama 3.3 70B)")
        embed = Sagemaker("all-MiniLM-L6-v2\n(local, 80 MB)")

    # ── LangGraph Agent Pipeline ──────────────────────────────────────────────
    with Cluster("LangGraph Agent Pipeline (WS6) – Sequential StateGraph",
                 graph_attr={"bgcolor": "#e8eaf6", "fontcolor": "#283593"}):

        with Cluster("WS1 – Ingestion",
                     graph_attr={"bgcolor": "#e0f2f1", "fontcolor": "#004d40"}):
            scraper = Python("ScraperAgent\n(ws1_run_scraper.py)")

        with Cluster("WS2 – Sponsorship",
                     graph_attr={"bgcolor": "#fff3e0", "fontcolor": "#bf360c"}):
            sponsor = Python("SponsorshipAgent\n(ws2_build_sponsorship.py)")

        with Cluster("WS3 – Resume Parsing",
                     graph_attr={"bgcolor": "#e3f2fd", "fontcolor": "#0d47a1",
                                 "rankdir": "TB"}):
            parser   = Python("ResumeParserAgent\n(ws3_resume_parser.py)")
            fallback = Python("Regex Fallback\nParser")

        with Cluster("WS4 – Matching",
                     graph_attr={"bgcolor": "#f3e5f5", "fontcolor": "#4a148c",
                                 "rankdir": "TB"}):
            indexer = Python("IndexBuilderAgent\n(ws4_job_matcher.py)")
            matcher = Python("MatchingAgent\nCosine Similarity")

        with Cluster("WS5 – Scoring",
                     graph_attr={"bgcolor": "#fce4ec", "fontcolor": "#880e4f",
                                 "rankdir": "TB"}):
            signal   = Python("SignalExtractAgent\n(JD regex signals)")
            evidence = Python("HistoricalEvidenceAgent\n(PERM / USCIS lookup)")
            auditor  = Python("AuditorAgent\n(Cross-validate & rank)")

    # ── Storage / Medallion Layer ─────────────────────────────────────────────
    with Cluster("Storage – Medallion Architecture",
                 graph_attr={"bgcolor": "#fff8e1", "fontcolor": "#e65100",
                             "rankdir": "TB"}):
        bronze   = Storage("Bronze Layer\n(Raw JSON jobs)")
        silver   = Storage("Silver Layer\n(Deduplicated Parquet)")
        spons_db = Storage("sponsorship_history\n.parquet")
        job_idx  = Storage("Job Embeddings Index\n(NumPy / Parquet)")
        cand_emb = Storage("Candidate\nEmbeddings")

    # ── Scoring & Data Models ─────────────────────────────────────────────────
    with Cluster("Scoring & Data Models",
                 graph_attr={"bgcolor": "#ede7f6", "fontcolor": "#311b92",
                             "rankdir": "TB"}):
        scorer     = Python("ConfidenceScorer\nmatch 40%·visa 35%·recency 25%")
        scored_job = Python("ScoredJob\n(job_id · score · reasoning)")
        cand_prof  = Python("CandidateProfile\n(skills · exp · visa_status)")
        job_match  = Python("JobMatch\n(similarity · skill_overlap)")

    # ═════════════════════════════════════════════════════════════════════════
    # EDGES
    # ═════════════════════════════════════════════════════════════════════════

    # User ↔ Application Layer
    user     >> Edge(color="#1565c0", label="opens UI")    >> frontend
    backend  >> Edge(color="#1565c0", label="results")     >> frontend
    frontend >> Edge(color="#1565c0", label="ranked jobs") >> user

    # Backend triggers resume parsing
    backend >> Edge(color="#1565c0", label="trigger") >> parser

    # Data sources → ingestion agents
    ats   >> Edge(color="#00897b", label="job listings")  >> scraper
    perm  >> Edge(color="#ef6c00", label="cert. filings") >> sponsor
    uscis >> Edge(color="#ef6c00", label="H-1B records")  >> sponsor

    # Ingestion → medallion storage
    scraper >> Edge(color="#f9a825", label="raw JSON")       >> bronze
    bronze  >> Edge(color="#f9a825", label="dedup & enrich") >> silver
    sponsor >> Edge(color="#f9a825", label="parquet")        >> spons_db

    # Storage → downstream agents (dashed = read)
    silver   >> Edge(color="#546e7a", style="dashed") >> indexer
    spons_db >> Edge(color="#546e7a", style="dashed") >> evidence

    # External AI: Groq parse
    parser >> Edge(color="#c62828", label="parse req")                    >> groq
    groq   >> Edge(color="#c62828", label="parsed JSON", style="dashed")  >> parser
    parser >> Edge(color="#c62828", label="fallback",    style="dashed")  >> fallback

    # LangGraph sequential pipeline flow
    scraper  >> Edge(color="#1565c0") >> sponsor
    sponsor  >> Edge(color="#1565c0") >> parser
    parser   >> Edge(color="#1565c0") >> indexer
    indexer  >> Edge(color="#1565c0") >> matcher
    matcher  >> Edge(color="#6a1b9a") >> signal
    signal   >> Edge(color="#6a1b9a") >> evidence
    evidence >> Edge(color="#6a1b9a") >> auditor

    # Embedding model
    indexer >> Edge(color="#00695c", style="dashed")         >> embed
    embed   >> Edge(color="#00695c", label="job vectors")    >> job_idx
    embed   >> Edge(color="#00695c", label="resume vectors") >> cand_emb

    # Embedding indexes → matcher
    job_idx  >> Edge(color="#546e7a", style="dashed") >> matcher
    cand_emb >> Edge(color="#546e7a", style="dashed") >> matcher

    # Scoring chain
    auditor >> Edge(color="#6a1b9a")                     >> scorer
    scorer  >> Edge(color="#6a1b9a")                     >> scored_job
    matcher >> Edge(color="#6a1b9a", label="match data") >> cand_prof
    matcher >> Edge(color="#6a1b9a")                     >> job_match

    # Results → backend
    scored_job >> Edge(color="#2e7d32", label="ranked jobs") >> backend
