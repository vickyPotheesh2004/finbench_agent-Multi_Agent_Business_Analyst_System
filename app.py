"""
app.py
FinBench Multi-Agent Business Analyst AI — Streamlit UI
PDR-BAAAI-001 · Rev 1.0

3-Screen Interface:
    1. UPLOAD      — document upload + metadata
    2. STATUS      — live analysis progress
    3. RESULTS     — answer + confidence + download DOCX

Launch:
    streamlit run app.py

Constraints:
    C1  $0 cost — Streamlit free
    C2  100% local — no external network
    C9  _rlef_ fields NEVER shown in UI
"""

import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "FinBench Analyst AI",
    page_icon  = "📊",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a4d7a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1a4d7a;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .status-running   { color: #f0a020; font-weight: 600; }
    .status-complete  { color: #2d862d; font-weight: 600; }
    .status-pending   { color: #999;    font-weight: 600; }
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #1a4d7a;
        padding: 1.5rem;
        border-radius: 4px;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .confidence-high   { color: #2d862d; font-weight: 600; }
    .confidence-medium { color: #f0a020; font-weight: 600; }
    .confidence-low    { color: #cc3333; font-weight: 600; }
    .hitl-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state():
    defaults = {
        "screen":         "upload",
        "document_path":  "",
        "company_name":   "",
        "doc_type":       "10-K",
        "fiscal_year":    f"FY{datetime.now().year}",
        "session_id":     "",
        "ingestion_done": False,
        "ba_state":       None,
        "current_query":  "",
        "answer":         "",
        "confidence":     0.0,
        "report_path":    "",
        "risk_score":     0.0,
        "progress":       0,
        "current_node":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_rlef(d: dict) -> dict:
    """C9 enforcement — remove any _rlef_ fields from display."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if not k.startswith("_rlef_")}


def _confidence_class(conf: float) -> str:
    if conf >= 0.85: return "confidence-high"
    if conf >= 0.60: return "confidence-medium"
    return "confidence-low"


def _confidence_label(conf: float) -> str:
    if conf >= 0.85: return "HIGH"
    if conf >= 0.60: return "MEDIUM"
    return "LOW"


def _check_ollama() -> dict:
    """Check if Ollama/Llama 3.1 8b is running."""
    try:
        from src.utils.llm_client import get_llm_client
        client = get_llm_client()
        return client.health_check()
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to documents/ folder and return path."""
    docs_dir = ROOT / "documents"
    docs_dir.mkdir(exist_ok=True)
    path = docs_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)


def _run_ingestion(document_path: str, metadata: dict) -> dict:
    """Run N01-N03 ingestion pipeline (with optional N01b image processing)."""
    try:
        from src.ingestion.pdf_ingestor         import PDFIngestor
        from src.ingestion.section_tree_builder import SectionTreeBuilder
        from src.ingestion.chunker              import Chunker
        from src.state.ba_state                 import BAState

        # Optional Llama 3.1 8b client for chart vision
        llm_client = None
        if metadata.get("enable_images"):
            try:
                from src.utils.llm_client import get_llm_client
                llm_client = get_llm_client()
            except Exception:
                pass

        state = BAState(
            session_id    = metadata["session_id"],
            document_path = document_path,
            company_name  = metadata["company_name"],
            doc_type      = metadata["doc_type"],
            fiscal_year   = metadata["fiscal_year"],
        )

        ingestor = PDFIngestor(
            enable_images = metadata.get("enable_images", False),
            llm_client    = llm_client,
        )
        state = ingestor.run(state)
        state = SectionTreeBuilder().run(state)
        state = Chunker().run(state)

        return {"success": True, "state": state}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _run_query(state, question: str) -> dict:
    """Run N04-N19 query pipeline."""
    try:
        from src.pipeline.pipeline import FinBenchPipeline
        pipeline = FinBenchPipeline()
        result = pipeline.query(state, question)
        return {"success": True, "state": result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Header + system status ────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">FinBench Analyst AI</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">'
                'Multi-Agent Business Analyst · Llama 3.1 8b · 100% Local · $0 Cost'
                '</div>', unsafe_allow_html=True)

with col_h2:
    health = _check_ollama()
    if health.get("available"):
        st.success("🟢 Llama 3.1 8b Online")
    else:
        st.warning("🟡 Llama 3.1 8b Offline")
        st.caption("Start: `ollama serve`")


# ── SIDEBAR — Navigation + Document info ──────────────────────────────────────

with st.sidebar:
    st.markdown("### 📁 Current Document")

    if st.session_state.ingestion_done:
        st.success("✅ Document Ingested")
        st.write(f"**Company:** {st.session_state.company_name}")
        st.write(f"**Type:** {st.session_state.doc_type}")
        st.write(f"**FY:** {st.session_state.fiscal_year}")
        if st.session_state.ba_state:
            st.write(f"**Chunks:** {st.session_state.ba_state.chunk_count}")

        st.divider()
        if st.button("🔄 New Document", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _init_state()
            st.rerun()
    else:
        st.info("📤 Upload a document to begin")

    st.divider()
    st.markdown("### 🎯 System Status")

    if st.session_state.ingestion_done:
        st.markdown('<span class="status-complete">✅ Ingestion</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pending">⏳ Ingestion</span>',
                    unsafe_allow_html=True)

    if st.session_state.answer:
        st.markdown('<span class="status-complete">✅ Analysis</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pending">⏳ Analysis</span>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### ℹ️ Constraints")
    st.caption("C1: $0 cost · C2: 100% local · C5: seed=42")


# ── SCREEN 1 — UPLOAD ─────────────────────────────────────────────────────────

if st.session_state.screen == "upload":
    st.markdown("### 📤 Upload Financial Document")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Drop PDF, DOCX, XLSX, CSV or HTML",
            type = ["pdf", "docx", "xlsx", "csv", "html", "txt"],
            help = "10-K, 10-Q, 8-K, earnings transcripts, or research reports",
        )

        if uploaded:
            st.info(f"📄 **{uploaded.name}** · "
                    f"{uploaded.size / 1024:.0f} KB")

    with col2:
        st.markdown("**Document Metadata**")
        company = st.text_input("Company Name",
                                value       = st.session_state.company_name,
                                placeholder = "Apple Inc")
        doc_type = st.selectbox(
            "Document Type",
            options = ["10-K", "10-Q", "8-K", "Earnings Call",
                       "Investor Presentation", "Other"],
            index   = 0,
        )
        fiscal_year = st.text_input("Fiscal Year",
                                    value = st.session_state.fiscal_year)

        enable_images = st.checkbox(
            "🖼 Enable OCR + Chart Vision",
            value = False,
            help  = "Slower but extracts data from scanned pages and charts. Requires Tesseract + Llama 3.1 8b.",
        )

    st.divider()

    ready = uploaded and company and fiscal_year
    if st.button("🚀 Analyse Document",
                 type              = "primary",
                 disabled          = not ready,
                 use_container_width = True):

        # Save uploaded file
        doc_path = _save_uploaded_file(uploaded)
        st.session_state.document_path = doc_path
        st.session_state.company_name  = company
        st.session_state.doc_type      = doc_type
        st.session_state.fiscal_year   = fiscal_year
        st.session_state.session_id    = str(uuid.uuid4())

        # Run ingestion
        spinner_msg = ("Ingesting with OCR + Vision (slower)..."
                       if enable_images else "Ingesting document...")
        with st.spinner(spinner_msg):
            result = _run_ingestion(doc_path, {
                "company_name":  company,
                "doc_type":      doc_type,
                "fiscal_year":   fiscal_year,
                "session_id":    st.session_state.session_id,
                "enable_images": enable_images,
            })

        if result["success"]:
            st.session_state.ba_state       = result["state"]
            st.session_state.ingestion_done = True
            st.session_state.screen         = "analysis"
            st.success(f"✅ Ingested {result['state'].chunk_count} chunks")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ Ingestion failed: {result['error']}")


# ── SCREEN 2 — ANALYSIS + QUERY ──────────────────────────────────────────────

elif st.session_state.screen == "analysis":
    st.markdown(f"### 📊 {st.session_state.company_name} — "
                f"{st.session_state.doc_type} {st.session_state.fiscal_year}")

    # Show ingestion metrics
    if st.session_state.ba_state:
        m1, m2, m3, m4 = st.columns(4)
        state = st.session_state.ba_state
        with m1:
            st.metric("Chunks",
                      getattr(state, "chunk_count", 0))
        with m2:
            section_tree = getattr(state, "section_tree", {}) or {}
            st.metric("Sections",
                      section_tree.get("total_sections", 0))
        with m3:
            st.metric("BM25 Index",
                      "Built" if getattr(state, "bm25_index_path", "") else "—")
        with m4:
            st.metric("ChromaDB",
                      "Built" if getattr(state, "chromadb_collection", "") else "—")

    st.divider()

    # Query input
    st.markdown("### 🔍 Ask a Question")

    example_queries = [
        "What was total net sales for FY2023?",
        "What was net income for the most recent fiscal year?",
        "Calculate the diluted EPS and verify against the income statement.",
        "What are the top 3 risk factors?",
        "Summarize the MD&A section.",
    ]

    q_col1, q_col2 = st.columns([4, 1])
    with q_col1:
        question = st.text_input(
            "Enter your financial question:",
            value       = st.session_state.current_query,
            placeholder = example_queries[0],
        )
    with q_col2:
        st.write("")
        st.write("")
        run_btn = st.button("▶ Run Analysis",
                            type              = "primary",
                            use_container_width = True)

    with st.expander("💡 Example questions"):
        for ex in example_queries:
            if st.button(f"→ {ex}", key=f"ex_{hash(ex)}"):
                st.session_state.current_query = ex
                st.rerun()

    if run_btn and question:
        st.session_state.current_query = question

        # Progress placeholder
        progress_bar = st.progress(0, text="Initialising...")
        status_text  = st.empty()

        # Run query
        nodes = [
            ("N04 CART Router",      10),
            ("N05 Difficulty",       15),
            ("N06 SniperRAG",        25),
            ("N07 BM25",             35),
            ("N08 BGE-M3",           45),
            ("N09 RRF+Reranker",     55),
            ("N10 Prompt Assembler", 62),
            ("N11 Analyst Pod",      70),
            ("N12 CFO/Quant Pod",    78),
            ("N13 TriGuard",         82),
            ("N14 Blind Auditor",    86),
            ("N15 PIV Mediator",     90),
            ("N16 SHAP+DAG",         93),
            ("N18 RLEF Grading",     96),
            ("N19 DOCX Report",      99),
        ]

        # Animated progress while query runs
        for node_name, pct in nodes[:6]:
            progress_bar.progress(pct / 100, text=f"Running {node_name}...")
            time.sleep(0.2)

        with st.spinner("Running full analysis pipeline..."):
            result = _run_query(st.session_state.ba_state, question)

        if result["success"]:
            state = result["state"]

            # Complete progress
            for node_name, pct in nodes[6:]:
                progress_bar.progress(pct / 100, text=f"Completed {node_name}")
                time.sleep(0.05)
            progress_bar.progress(1.0, text="✅ Analysis complete")

            st.session_state.answer      = state.final_answer
            st.session_state.confidence  = state.confidence_score
            st.session_state.report_path = state.final_report_path or ""
            st.session_state.risk_score  = getattr(state, "risk_score", 0.0)
            st.session_state.ba_state    = state
            st.session_state.screen      = "results"
            time.sleep(0.5)
            st.rerun()
        else:
            progress_bar.empty()
            st.error(f"❌ Analysis failed: {result['error']}")


# ── SCREEN 3 — RESULTS + DOWNLOAD ────────────────────────────────────────────

elif st.session_state.screen == "results":
    st.markdown(f"### 📊 Analysis Results")
    st.caption(f"Query: _{st.session_state.current_query}_")

    # Confidence badge
    conf       = st.session_state.confidence
    conf_class = _confidence_class(conf)
    conf_label = _confidence_label(conf)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:0.9rem;color:#666">Confidence</div>'
            f'<div class="{conf_class}" style="font-size:1.8rem">'
            f'{conf:.2f} {conf_label}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        risk = st.session_state.risk_score
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:0.9rem;color:#666">Risk Score</div>'
            f'<div style="font-size:1.8rem;color:#1a4d7a">{risk:.1f}/100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        state     = st.session_state.ba_state
        sniper    = getattr(state, "sniper_hit", False) if state else False
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:0.9rem;color:#666">SniperRAG Hit</div>'
            f'<div style="font-size:1.8rem;color:#1a4d7a">'
            f'{"YES" if sniper else "NO"}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        pod = getattr(state, "winning_pod", "—") if state else "—"
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:0.9rem;color:#666">Winning Pod</div>'
            f'<div style="font-size:1.3rem;color:#1a4d7a;padding-top:0.3rem">'
            f'{pod}</div></div>',
            unsafe_allow_html=True,
        )

    # Low-confidence warning (HITL trigger)
    if getattr(state, "low_confidence", False):
        st.markdown(
            '<div class="hitl-warning">'
            '⚠ <b>HITL Review Recommended</b> — Confidence is below threshold. '
            'Verify the answer against source documents before use.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Answer
    st.markdown("### 💬 Answer")
    st.markdown(
        f'<div class="answer-box">{st.session_state.answer}</div>',
        unsafe_allow_html=True,
    )

    # Citations
    if state:
        citations = (
            getattr(state, "analyst_citations", [])
            or getattr(state, "quant_citations",   [])
            or []
        )
        if citations:
            with st.expander("📎 Citations"):
                for c in citations[:5]:
                    st.write(f"• {c}")

    # Chart data from image vision (N01b)
    if state:
        chart_cells = [
            c for c in (getattr(state, "table_cells", []) or [])
            if isinstance(c, dict) and c.get("source") == "chart_vision"
        ]
        if chart_cells:
            with st.expander(f"📊 Chart Data from Images ({len(chart_cells)})"):
                for cell in chart_cells[:20]:
                    unit = f" {cell.get('unit', '')}" if cell.get("unit") else ""
                    st.write(
                        f"• **{cell.get('label')}**: {cell.get('value')}{unit} "
                        f"(page {cell.get('page')}, "
                        f"{cell.get('chart_type', 'chart')})"
                    )

# Download PDF report (lazy — generates on first click only)
    st.divider()
    st.markdown("### 📥 Download Professional PDF Report")
    st.caption(
        "14-page business analyst grade report with executive summary, "
        "answer card with confidence gauge, reasoning chain, 4-tier "
        "retrieval evidence, 3-pod comparison, Benford forensic chart, "
        "risk gauge, SHAP explainability, Causal DAG, chart data from "
        "images, methodology, citations appendix, and validator audit trail."
    )

    # Lazy generation — only build PDF when requested
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_path  = None

    if st.session_state.pdf_bytes is None:
        if st.button("🛠 Generate PDF Report",
                     use_container_width=True, type="primary"):
            try:
                from src.output.pdf_report_generator import PDFReportGenerator
                with st.spinner("Generating 14-page PDF with charts... (~5 sec)"):
                    pdf_gen  = PDFReportGenerator()
                    pdf_path = pdf_gen.generate(st.session_state.ba_state)
                with open(pdf_path, "rb") as fp:
                    st.session_state.pdf_bytes = fp.read()
                st.session_state.pdf_path = pdf_path
                st.rerun()
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
                st.info("If reportlab is missing, run: "
                        "`pip install reportlab matplotlib`")
    else:
        kb = len(st.session_state.pdf_bytes) // 1024
        st.success(
            f"📄 Ready: `{os.path.basename(st.session_state.pdf_path)}` "
            f"({kb} KB · 14 pages)"
        )
        st.download_button(
            "⬇ Download PDF Report",
            data                = st.session_state.pdf_bytes,
            file_name           = os.path.basename(st.session_state.pdf_path),
            mime                = "application/pdf",
            use_container_width = True,
            type                = "primary",
        )

    # Actions
    st.divider()
    a1, a2 = st.columns(2)
    with a1:
        if st.button("🔍 Ask Another Question", use_container_width=True):
            st.session_state.answer     = ""
            st.session_state.confidence = 0.0
            st.session_state.pdf_bytes  = None
            st.session_state.pdf_path   = None
            st.session_state.screen     = "analysis"
            st.rerun()
    with a2:
        if st.button("📤 Upload New Document", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "FinBench Multi-Agent Business Analyst AI · PDR-BAAAI-001 Rev 1.0 · "
    "Llama 3.1 8b local · No document ever leaves your machine"
)