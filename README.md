
<div align="center">

# 🚀 FinBench Multi-Agent Business Analyst AI

<img src="./finbench-github-banner.png" alt="FinBench Banner" width="100%"/>

</div>

---


<div align="center">

# 📊 FinBench Multi-Agent Business Analyst AI

### Intelligent Financial Reasoning. Local Execution. Zero API Cost.

> A reproducible, fully local, multi-agent AI system that extracts, validates, debates, and explains financial insights directly from SEC 10-K filings using a 19-node intelligent pipeline architecture.

<br>

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Tests](https://img.shields.io/badge/Tests-128%2B_Passing-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-84.2%25-brightgreen.svg)
![Local AI](https://img.shields.io/badge/100%25-Local_AI-purple.svg)
![Cost](https://img.shields.io/badge/API_Cost-$0-success.svg)

</div>

---

# 🌟 Overview

FinBench AI is a next-generation **multi-agent financial analysis platform** designed to process SEC 10-K filings with high accuracy using local AI infrastructure.

Unlike traditional RAG systems, this architecture combines:

- ⚡ Hybrid Retrieval
- 🧠 Multi-Agent Debate
- 📈 Financial Reasoning
- 🔍 Explainability
- 🛡️ Validation Layers
- 📄 Structured Financial Grounding

All running completely **offline** on consumer hardware.

---

# 🎯 Core Capability

Ask complex financial questions in natural language:

```text
"What was Apple's net income in FY2023?"
```

### ✅ Output

```yaml
Answer:
96,995 million

Confidence:
1.00

Source:
us-gaap:NetIncomeLoss

Reference:
[Apple Inc./10-K/FY2023/iXBRL_NUMERIC/24]

Execution Time:
0.1 seconds
```

---

# 🧠 Key Features

| Feature | Description |
|---|---|
| 🔥 SniperRAG Retrieval | Direct iXBRL table-cell extraction |
| 🧩 Hybrid Search | BM25 + Semantic + Reranker |
| 🤖 Multi-Agent Debate | Analyst vs CFO vs Blind Auditor |
| 🛡️ Validation Gates | Grounded reasoning validation |
| 📊 Explainability | SHAP + causal DAG analysis |
| 💾 100% Local | No cloud APIs required |
| ⚡ Fast Execution | Average 1.9 sec/query |
| 📄 Provenance Tracking | Every answer fully cited |
| 🔁 Graceful Degradation | Never crashes under failures |
| 🧪 Benchmark Tested | 128+ regression tests |

---

# 📈 Benchmark Results — Phase 1 (May 2026)

## 🏆 Overall Accuracy: **84.2%**

### Tested on:
- 165 numerical financial questions
- 7 major SEC 10-K filings
- Real-world financial extraction tasks

---

## 📊 Company-wise Performance

| Company | Filing | Accuracy | Notes |
|---|---|---|---|
| 🍎 Apple Inc. | FY2023 | **100.0%** (24/24) | Perfect iXBRL extraction |
| 🟢 Nvidia | FY2024 | 91.3% | Strong structured tables |
| 🛒 Amazon | FY2023 | 87.0% | High retrieval quality |
| 🔵 Alphabet | FY2023 | 83.3% | Complex reporting structures |
| 📘 Meta Platforms | FY2023 | 79.2% | Missing inventory sections |
| 🪟 Microsoft | FY2023 | 75.0% | Multi-period ambiguity |
| 🚗 Tesla | FY2023 | 73.9% | Smaller iXBRL footprint |

---

# 🏗️ Complete System Architecture

```text
INGESTION → ROUTING → HYBRID RETRIEVAL → MULTI-AGENT DEBATE → EXPLAINABILITY → OUTPUT
```

---

# 🔥 Architectural Innovations

## 1️⃣ Sniper-First Retrieval Strategy

Traditional RAG systems waste compute.

FinBench uses **SniperRAG** first:
- ⚡ 0.1–0.3 sec response
- 🎯 High numerical precision
- 💰 Zero API dependency
- 🧠 Minimal hallucination

---

## 2️⃣ Multi-Agent Debate System

| Agent | Role |
|---|---|
| 🧠 Analyst Pod | Financial reasoning |
| 💼 CFO/Quant Pod | Numerical validation |
| 🕵️ Blind Auditor | Detects hallucinations/groupthink |

---

# 🚀 Quick Start

## 📦 Installation

```bash
git clone https://github.com/vickyPotheesh2004/finbench_agent-Multi_Agent_Business_Analyst_System.git

cd finbench_agent-Multi_Agent_Business_Analyst_System

python -m venv venv

source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🧠 Install Local LLM (Optional)

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2.5:3b
```

---

## ▶️ Run Example

```python
from src.pipeline.pipeline import FinBenchPipeline

pipeline = FinBenchPipeline()

state = pipeline.ingest(
    "documents/sec_filings/AAPL_FY2023_10-K.html",
    company_name="Apple Inc.",
    doc_type="10-K",
    fiscal_year="FY2023"
)

state = pipeline.query(
    state,
    "What was Apple revenue in FY2023?"
)

print(state.final_answer)
```

---

# 🧪 Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangGraph |
| Vector DB | ChromaDB |
| Semantic Retrieval | BGE-M3 |
| Sparse Retrieval | BM25 |
| LLM Runtime | Ollama |
| Models | Qwen2.5, Llama 3.1 |
| ML | XGBoost, sklearn |
| Explainability | SHAP |
| Graph Analysis | networkx |
| Visualization | matplotlib, plotly |
| Export | python-docx |
| Database | SQLite |
| Testing | pytest |

---

# 📍 Future Roadmap

| Phase | Objective | Target |
|---|---|---|
| Phase 2 | Fix wrong-cell selection | 92–95% |
| Phase 3 | Official FinanceBench evaluation | Benchmark parity |
| Phase 4 | SFT + DPO fine-tuning | Higher reasoning |
| Phase 5 | Power BI & Tableau integration | Visual analytics |
| Phase 6 | Live RLEF self-improvement loop | Continual learning |

---

# 📜 License

MIT License

---

# 🙏 Acknowledgments

Inspired by:
- FinanceBench
- FinQA
- Patronus AI
- Qwen
- Llama
- BGE-M3 community

---

# 👨‍💻 Author

## Potheesh Vignesh K

- GitHub: github.com/vickyPotheesh2004
- LinkedIn: linkedin.com/in/vickypotheesh

---

<div align="center">

# ⭐ FinBench AI

### Local Financial Intelligence Meets Multi-Agent Reasoning

*"The future of financial analysis is not just AI.  
It is grounded, explainable, reproducible AI."*

</div>
