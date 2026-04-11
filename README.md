<p align="center">
  <strong>Lumisift</strong><br>
  <em>Your RAG pipeline loses 64% of scientific data. Lumisift keeps 87%.</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0"></a>
  <a href="#the-benchmark"><img src="https://img.shields.io/badge/validated-1%2C077_PubMed_articles-orange?style=flat-square" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/runs_locally-no_GPU_needed-brightgreen?style=flat-square" alt="No GPU"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#the-benchmark">Benchmarks</a> · <a href="#who-this-is-for">Use Cases</a> · <a href="#the-value">Value</a> · <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <sub>Created by <a href="https://www.linkedin.com/in/ben-moradtalab-9442a41a6">Saeed Moradtalab</a></sub>
</p>

---

## The Problem

A researcher asks your RAG system: *"What was the IC50 of compound LX-4291?"*

The AI hallucinates an answer. Not because the model is bad -- but because the **retrieval selected the wrong paragraph**. It picked the introduction (*"EGFR inhibitors have shown promise..."*) instead of the results (*"IC50 = 3.2 nM, 47-fold selectivity"*).

This happens because every RAG system selects context by **semantic similarity** -- *"which text sounds like the query?"* In science, the paragraph that sounds relevant rarely contains the actual data.

We measured this on 1,077 PubMed articles:

> **Standard retrieval discards 64% of numerical facts, 61% of comparative claims, and 59% of causal relationships before the LLM ever sees them.**

In pharma, a missing IC50 = a missed drug candidate. In clinical research, a lost p-value = a flawed review. In regulatory work, a dropped threshold = a compliance failure.

**Lumisift sits between your retrieval and your LLM and protects the data-rich paragraphs.**

---

## Quick Start

```bash
git clone https://github.com/Saeedmora/Lumisift.git
cd Lumisift
pip install -e .
python app.py        # Web UI at http://localhost:5000
```

**Python API:**

```python
from core.pipeline import LogicalRoomsPipeline

pipe = LogicalRoomsPipeline()
result = pipe.select_context(
    chunks,
    query="What is the IC50 of LX-4291?",
    mode="hybrid", alpha=0.3, top_k=5
)
# result.selected_chunks → send to your LLM
# 49% fewer tokens, 85% of data preserved
```

**Diagnose your own pipeline:**

```bash
python information_loss_taxonomy.py   # What does YOUR retrieval lose?
```

> [!TIP]
> No API keys needed. Everything runs 100% locally. No data leaves your machine.

---

## How It Works

Lumisift adds **one scoring step** to your pipeline: information density detection. Chunks with numbers, measurements, comparisons, and experimental results get boosted before selection.

```
Standard:   Document → Chunk → Embed → Rank by similarity → LLM
                                              ↓
                              "Sounds like the query" ✓
                              "Contains actual data"  ✗

Lumisift:   Document → Chunk → Embed → Rank by similarity + density → LLM
                                              ↓
                              "Sounds like the query" ✓
                              "Contains IC50, p-values, fold-changes" ✓✓
```

No new model. No cloud API. No GPU. One mechanism that preserves the data your researchers need.

---

## Who This Is For

<table>
<tr>
<td width="50%">

### 🧬 Pharmaceutical R&D

Your AI summarizes background instead of results. Medicinal chemists lose trust.

**With Lumisift:** Standard RAG retained **15%** of critical drug data across 3 pharma scenarios. Lumisift retained **84%**.

</td>
<td width="50%">

### 🧪 Biotech & Protein Engineering

The AI can't find kinetic data (kcat/Km, E-values). It selected the methods section instead of results.

**With Lumisift:** Embedding retrieval kept **0 of 7** kinetic parameters. Lumisift kept **6 of 7**.

</td>
</tr>
<tr>
<td>

### 📊 Clinical Research

Trial results are missing p-values, hazard ratios, confidence intervals. The statistical evidence is silently discarded.

**With Lumisift:** p-value retention: 34% → **91%**. The statistical backbone survives.

</td>
<td>

### 📋 Regulatory & Academic

Auditors and PhDs need exact values, not paraphrases. The AI approximates when it should quote.

**With Lumisift:** Exact measurements, concentrations, and thresholds get selection priority.

</td>
</tr>
</table>

---

## The Benchmark

All results validated on **1,077 PubMed articles** across 10 biomedical domains, **6,463 text chunks**, **2,722 numerical facts**. Every script is included. Every result is reproducible.

<details>
<summary><strong>📂 Corpus breakdown (10 domains, click to expand)</strong></summary>

| Domain | ~Articles | Example Data |
|--------|----------|-------------|
| Protein engineering | 120 | kcat/Km, thermostability, fold-change |
| Drug discovery | 120 | IC50, SAR, selectivity ratios |
| Enzyme optimization | 110 | Activity, enantioselectivity (E-value) |
| Antibody engineering | 110 | Kd, affinity maturation |
| Pharmacokinetics | 110 | ADME, bioavailability, half-life |
| Vaccine development | 107 | Neutralizing titers, efficacy |
| mRNA delivery | 100 | Transfection efficiency, LNP size |
| CRISPR gene editing | 100 | Knock-out efficiency, off-target |
| Biocatalysis | 100 | Conversion rates, TON |
| Protein extraction | 100 | Yields, purity, recovery |

</details>

### Lumisift vs. 6 retrieval baselines

| Method | Data Retained | Delta |
|--------|:------------:|:-----:|
| Embedding (MiniLM) | 38% | — |
| BM25 (keyword) | 42% | +4pp |
| ColBERT (token-level) | 44% | +6pp |
| Cross-Encoder (ms-marco) | 44% | +6pp |
| **Lumisift (heuristic)** | **83%** | **+45pp** |
| **Lumisift (learned)** | **87%** | **+49pp** |

> [!IMPORTANT]
> Four fundamentally different architectures — keyword, dense, late-interaction, cross-encoder — all lose 56-62% of quantitative data. The method of matching doesn't matter. **Matching doesn't know what data is.** Lumisift does.

### What types of information get lost?

| Type | Embedding loses | Lumisift keeps | Recoverable? |
|------|:--------------:|:--------------:|:------------:|
| 📐 Numerical facts | 64% | **85%** | ✅ Yes (+49pp) |
| ⚖️ Comparative claims | 61% | **87%** | ✅ Yes (+48pp) |
| 🔗 Causal statements | 59% | 50% | ⚠️ Partial (+9pp) |
| ❓ Uncertainty markers | 54% | 49% | ❌ No |
| 🔬 Methods details | 40% | 57% | ❌ No (-3pp) |
| 🏷️ Named entities | 35% | 71% | ⚠️ Partial (+5pp) |

Numbers and comparisons = fully recoverable. Methods and uncertainty = not. That's a design trade-off, not a bug.

### The Power of Specificity

We ablated every component to find what drives the result:

| Test | Retention | Takeaway |
|------|:---------:|---------|
| Specificity detection alone | **90%** | Outperforms the full system |
| Full system (8 axes) | 83% | Baseline |
| Remove specificity | 48% | Collapses to embedding-level |
| Remove any other axis | 83% | Zero effect |

> [!NOTE]
> Lumisift's advantage comes from one signal: **information density detection** — finding chunks with numbers, measurements, and quantitative data. That's the contribution. We built seven other scoring axes for future use (trust, causality, temporal, etc.), but for numerical retention, specificity is the mechanism. We're transparent about this.

### Learned model replaces regex

The heuristic uses regex. Fragile, domain-specific. The learned model fixes both:

| | Heuristic | Learned Model |
|-|:---------:|:------------:|
| Retention | 83% | **87%** (+5pp) |
| Regex-dependent | Yes | **No** |
| Retrainable | No | **Yes, on your data** |
| Model size | — | **368 KB** |
| Speed | ~0ms | ~0.1ms/chunk |

---

## The Value

### 1. Your AI stops guessing at numbers

| | Without Lumisift | With Lumisift |
|-|-----------------|--------------|
| LLM output | *"The compound showed moderate activity"* | *"IC50 = 3.2 nM, 47-fold selectivity over wild-type"* |
| Source | Hallucinated summary | Actual data from paper |

### 2. You cut API costs in half

| Scale | Before | After | Monthly Savings |
|-------|:------:|:-----:|:--------------:|
| 100 papers/day | $5/day | $2.40/day | **$78** |
| 1K papers/month | $150 | $72 | **$78** |
| 10K papers/month | $1,500 | $720 | **$780** |

### 3. You see what your pipeline drops

Even without adopting Lumisift, the **loss taxonomy** diagnoses your retrieval system:

```bash
python information_loss_taxonomy.py
# → "Your system silently discards 64% of numerical facts."
# Now you know. Now you can act.
```

---

## Limitations

> [!WARNING]
> Read this before adopting. We document failure as carefully as success.

| Limitation | Detail |
|-----------|--------|
| **One mechanism** | Specificity alone outperforms the full system. You're adopting a data density detector, not "multi-axis intelligence." |
| **IC50 sample size** | "100% IC50 retention" is n=24. Could be dataset-specific. We report sample size with every claim. |
| **Comprehension drops** | PubMedQA: 46.7% vs 93.3% full text. Data chunks ≠ explanation chunks. Use hybrid mode for mixed queries. |
| **Methods text** | Lumisift deprioritizes procedural text by -3pp. Design trade-off: data wins over methods. |
| **Biomedical focus** | Trained on PubMed. Legal/financial text needs retraining via the learned model. |
| **No human validation** | AI-judged quality only. Expert evaluation planned. |

---

## Roadmap

| | Milestone | Status |
|-|-----------|:------:|
| ✅ | Information loss taxonomy (6 types, 1,070 articles) | Done |
| ✅ | Ablation study (specificity = the mechanism) | Done |
| ✅ | Learned utility model (87%, no regex, 368 KB) | Done |
| ✅ | 6-method baseline comparison (+42pp over cross-encoder) | Done |
| ✅ | Drug discovery validation (84% vs 15%) | Done |
| ✅ | Reproducibility kit (200 articles, verifiable) | Done |
| 🔜 | LangChain / LlamaIndex drop-in plugin | Next |
| 🔜 | Cross-domain transfer (legal, financial, clinical) | Planned |
| 🔜 | Multi-objective learning (separate targets per axis) | Planned |
| 🔜 | Human expert evaluation | Planned |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       LUMISIFT                          │
│                                                         │
│  Input:  Your text chunks (from any retrieval system)   │
│  Output: Reranked by information density                │
│                                                         │
│  Backends:          Modes:                              │
│   Heuristic ~0ms     lumisift   = max data retention    │
│   Learned   ~0.1ms   similarity = max semantic match    │
│   TinyLlama ~200ms   hybrid     = both (recommended)    │
└─────────────────────────────────────────────────────────┘
```

---

## Configuration

| Variable | Required | Purpose |
|----------|:--------:|---------|
| `GEMINI_API_KEY` | Optional | AI-judged quality evaluation |
| `HF_TOKEN` | Optional | Faster HuggingFace downloads |
| `MODEL_PATH` | Optional | Custom GGUF model path |

No API keys? Everything still runs. **100% local.**

---

<details>
<summary><strong>📁 Project Structure</strong></summary>

```
Lumisift/
├── Core
│   ├── core/pipeline.py                 # select_context() API
│   ├── core/axes_evaluator.py           # Scoring engine
│   ├── core/embeddings.py               # MiniLM embeddings
│   └── app.py                           # Flask web interface
│
├── Diagnostics
│   ├── information_loss_taxonomy.py     # What does retrieval lose?
│   ├── ablation_study.py                # Which component matters?
│   └── information_utility_model.py     # Train learned model
│
├── Benchmarks
│   ├── numerical_retention_benchmark.py
│   ├── baseline_comparison.py           # BM25 / ColBERT / Embedding
│   ├── cross_encoder_benchmark.py       # Cross-encoder reranker
│   ├── drug_discovery_usecase.py        # 3 pharma scenarios
│   ├── hybrid_benchmark.py              # Alpha sweep
│   ├── pubmed_benchmark.py              # 1,077-article corpus
│   ├── pubmedqa_benchmark.py            # Comprehension test
│   ├── downstream_eval.py              # AI-judged quality
│   └── export_reproducibility_kit.py    # Verifiable dataset
│
└── Data
    ├── benchmark_data/                  # Results (JSON)
    └── models/                          # Trained models (.pt)
```

</details>

---

## Contributing

**Highest-impact contributions right now:**

1. 🌐 **Cross-domain testing** — Run the loss taxonomy on legal, financial, or clinical text
2. 🔌 **LangChain/LlamaIndex plugin** — Make Lumisift a drop-in reranker
3. 👩‍🔬 **Human evaluation** — Domain expert ratings on selected vs full-text quality
4. 🧠 **Better utility signals** — Improve the learned model's training data

```bash
git clone https://github.com/Saeedmora/Lumisift.git
git checkout -b feature/your-improvement
python information_loss_taxonomy.py   # verify results
```

---

## License

**AGPL-3.0** — Free to use, modify, distribute. Source must be shared for deployed modifications. Attribution required.

**Commercial licensing** available. **Contact:** [Saeed Moradtalab](https://www.linkedin.com/in/ben-moradtalab-9442a41a6)

---

<p align="center">
  <strong>Lumisift</strong><br>
  Standard retrieval loses 64% of scientific data. Lumisift keeps 87%.<br>
  One signal. 1,077 articles. 6 baselines. Every result reproducible.<br>
  <sub>© 2026 Saeed Moradtalab</sub>
</p>
