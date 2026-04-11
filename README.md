<p align="center">
  <strong>Lumisift</strong><br>
  <em>Stop losing scientific data in your RAG pipeline.</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0"></a>
  <a href="#what-we-proved"><img src="https://img.shields.io/badge/validated-1077_PubMed_articles-orange?style=flat-square" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-not_required-brightgreen?style=flat-square" alt="No GPU"></a>
</p>

<p align="center">
  <sub>Created by <a href="https://www.linkedin.com/in/ben-moradtalab-9442a41a6">Saeed Moradtalab</a></sub>
</p>

---

## The Problem Lumisift Solves

Your team builds a RAG pipeline over scientific papers. A researcher asks: *"What was the IC50 of compound X?"*

The AI hallucinates an answer.

Not because the model is bad. Not because the paper isn't in the database. But because **your retrieval system selected the wrong paragraph**. It picked the introduction -- *"EGFR inhibitors have shown promise in NSCLC..."* -- instead of the results section -- *"IC50 = 3.2 nM, 47-fold selectivity over wild-type."*

This happens because every RAG system today selects context by **semantic similarity**: "which text looks like the query?" In science, the paragraph that *sounds* relevant rarely contains the actual data.

We measured this across 1,077 real PubMed articles. The numbers are bad:

> **Your retrieval system silently discards 64% of numerical facts, 61% of comparative claims, and 59% of causal relationships -- before the LLM ever sees them.**

That's not a minor quality issue. In pharmaceutical R&D, a missing IC50 means a missed drug candidate. In clinical research, a lost p-value means a flawed literature review. In regulatory work, a dropped concentration limit means a compliance failure.

**Lumisift fixes this.** It sits between your retrieval and your LLM, and makes sure the data-rich paragraphs don't get thrown away.

---

## How Lumisift Works

Lumisift adds one step to your RAG pipeline: **information density scoring**. Before your system picks which chunks go to the LLM, Lumisift checks each chunk for quantitative content -- numbers, measurements, comparisons, experimental results -- and boosts the priority of data-rich passages.

```
Your current pipeline:
  Document → Chunk → Embed → Rank by similarity → Send to LLM

With Lumisift:
  Document → Chunk → Embed → Rank by similarity + information density → Send to LLM
                                                    ↑
                                     "This chunk has IC50 values,
                                      fold-changes, and p-values.
                                      Keep it."
```

That's all it does. No new embedding model. No cloud API. No GPU. Just a smarter selection criterion that protects the data your researchers actually need.

---

## Three Ways to Use Lumisift

### 1. Drop-in Python API (most common)

```python
from core.pipeline import LogicalRoomsPipeline

pipe = LogicalRoomsPipeline()

# You already have chunks from your pipeline.
# Lumisift picks the ones that preserve data.
result = pipe.select_context(
    chunks,
    query="What is the IC50 of LX-4291?",
    mode="hybrid",     # Balances data preservation + semantic relevance
    alpha=0.3,         # 70% data priority, 30% similarity
    top_k=5
)

# Send result.selected_chunks to your LLM instead of all chunks.
# You just saved 49% of tokens AND kept 85% of the numbers.
```

### 2. Web Interface (for exploration)

```bash
python app.py
# Open http://localhost:5000
# Paste text → see how Lumisift scores each passage
# Compare what gets selected vs what gets dropped
```

### 3. Benchmark Your Own Data (for evaluation)

```bash
# See what YOUR retrieval system loses:
python information_loss_taxonomy.py    # Measures loss across 6 data types
python ablation_study.py               # Shows which component matters
```

---

## Who This Is For

### Pharmaceutical R&D Teams

**The problem you have:** Your AI assistant reads papers but can't find IC50 values, dosing data, or selectivity ratios. It summarizes background paragraphs instead of results. Medicinal chemists stop trusting the tool.

**What Lumisift does:** Ensures the results paragraph -- the one with *IC50 = 3.2 nM, tumor growth inhibition 89%, bioavailability 67%* -- gets selected over the introduction. In our three-scenario drug discovery benchmark, standard RAG retained **15%** of critical data. Lumisift retained **84%**.

### Biotech & Protein Engineering

**The problem you have:** An enzyme engineer asks the AI to compare variants. The AI can't see the kinetic data (kcat/Km, enantioselectivity, thermostability) because the retrieval picked the methods section instead of the results.

**What Lumisift does:** We tested this exact case. Embedding retrieval kept **zero** out of seven critical kinetic parameters for a lipase engineering paper. Lumisift kept **six**.

### Clinical Research & Literature Review

**The problem you have:** A clinical team reviews trial results, but the AI's context is missing p-values, hazard ratios, and confidence intervals. The statistical evidence that supports the conclusion is silently discarded.

**What Lumisift does:** p-value retention goes from 34% (standard RAG) to **91%** (Lumisift). The statistical backbone of the paper survives compression.

### Academic & Regulatory Work

**The problem you have:** A PhD student or regulatory auditor asks a specific quantitative question, and the AI approximates instead of quoting the exact value.

**What Lumisift does:** Exact measurements, concentrations, and thresholds get priority. The AI sees the actual number, not a paraphrase of it.

---

## What We Proved

Everything below is backed by reproducible benchmarks on **1,077 real PubMed articles** across 10 biomedical domains. Every script is included.

### The Benchmark Corpus

| Domain | ~Articles | Example Content |
|--------|----------|----------------|
| Protein engineering | 120 | Directed evolution, kcat/Km, thermostability |
| Drug discovery | 120 | IC50, SAR, in vivo efficacy, selectivity |
| Enzyme optimization | 110 | Activity, stability, enantioselectivity (E-value) |
| Antibody engineering | 110 | Affinity maturation, Kd, humanization |
| Pharmacokinetics | 110 | ADME, bioavailability, half-life, clearance |
| Vaccine development | 107 | Immunogenicity, neutralizing titers, efficacy |
| mRNA delivery | 100 | LNP formulation, transfection efficiency, expression |
| CRISPR gene editing | 100 | Knock-in/out efficiency, off-target rates |
| Biocatalysis | 100 | Industrial enzymes, conversion rates, TON |
| Protein extraction | 100 | Purification yields, purity, recovery |

After filtering: **1,070 articles** → **6,463 text chunks** → **2,722 numerical facts** tested.

---

### What Retrieval Methods Actually Lose

We tested 6 retrieval methods. The result surprised us: they all lose roughly the same amount of data, despite radically different architectures.

| Method | What it is | Data Retained | What this means |
|--------|-----------|--------------|-----------------|
| Embedding (MiniLM) | The default in most RAG systems | 38% | **62% of your numbers are gone** |
| BM25 | Keyword matching, the 20-year industry standard | 42% | Barely different from embeddings |
| ColBERT | State-of-the-art token-level matching | 44% | +2pp over BM25. Not meaningful. |
| Cross-Encoder | Joint query-document scoring, strongest reranker | 44% | +2pp over BM25. Still loses 56%. |
| **Lumisift (heuristic)** | **Information density detection (regex)** | **83%** | **Only 17% lost. 2x better.** |
| **Lumisift (learned)** | **Trained utility model (93K params)** | **87%** | **Best result. No regex.** |

**What this means for you:** It doesn't matter whether you use BM25, ColBERT, dense embeddings, or a cross-encoder. They all lose 56-62% of quantitative data. The architecture of matching doesn't help because *matching doesn't know what data is*. Lumisift does.

---

### What Types of Information Get Lost

Not all information is equally vulnerable. We measured 6 types:

| Information Type | What it is | How much embedding loses | How much Lumisift saves | Can you recover it? |
|-----------------|-----------|------------------------|----------------------|-------------------|
| **Numerical facts** | IC50 = 3.2 nM, 47-fold, p < 0.001 | **64% lost** | **85% kept** (+49pp) | YES |
| **Comparative claims** | "superior to", "3-fold improvement" | **61% lost** | **87% kept** (+48pp) | YES |
| **Causal statements** | "X inhibits Y", "leads to" | **59% lost** | **50% kept** (+9pp) | Partially |
| **Uncertainty markers** | "may suggest", "preliminary" | **54% lost** | **49% kept** (+3pp) | No |
| **Methodological details** | "using HPLC", "by PCR" | **40% lost** | **57% kept** (-3pp) | No (Lumisift worse) |
| **Named entities** | Protein names, drug names | **35% lost** | **71% kept** (+5pp) | Partially |

**What this means for you:** If your use case is about numbers and comparisons (drug discovery, clinical data, performance benchmarks), Lumisift dramatically reduces information loss. If your use case is about methodology or uncertainty language, Lumisift won't help -- and for methods, it actually makes things slightly worse.

This isn't a weakness we're hiding. It's a design consequence: prioritizing data-dense paragraphs means deprioritizing procedural ones. Know your use case.

---

### The Honest Result: One Mechanism Does the Work

We ran an ablation study -- removing each component one at a time to see what drives the results:

| What we tested | Result | What it means |
|---------------|--------|---------------|
| Full Lumisift system (8 scoring axes) | 83% retention | The default |
| **Specificity detection alone** | **90% retention** | **Better than the full system** |
| Remove specificity | 48% retention | System collapses to baseline |
| Remove any other axis | 83% retention | No change |

**What this means for you:** Lumisift's value comes from one thing: **detecting information density in text chunks**. The specificity mechanism -- finding passages with numbers, measurements, and quantitative data -- is the entire contribution. The other seven scoring axes (trust, causality, temporal, etc.) don't affect numerical retention. They exist for future extensions (comprehension, hedge detection, domain adaptation) but don't contribute to the core metric today.

We're transparent about this because it matters for your evaluation: you're adopting a **data density detector**, not a "multi-axis intelligence system." And that detector works.

---

### Learned Model: Beyond Regex

The heuristic version uses regex patterns to detect numbers. That's fragile and domain-specific. So we trained a neural replacement:

| | Heuristic (regex) | Learned Model |
|-|-------------------|---------------|
| Retention | 83% | **87%** |
| Regex-dependent? | Yes | **No** |
| Domain-transferable? | Manual effort | **Retrain on new data** |
| Model size | -- | **368 KB** |
| Speed | ~0ms | **~0.1ms** |

**What this means for you:** The learned model is the recommended path for production. It's the same idea (detect data-dense chunks) but generalized through training instead of hardcoded through regex. If you work with non-biomedical text, you can retrain it on your domain's data.

---

## The Value Proposition

At this stage, Lumisift delivers three concrete things:

### 1. Your AI stops hallucinating numbers.

When the LLM can see the actual IC50, the actual p-value, the actual fold-change -- it quotes them. When it can't see them, it guesses. Lumisift makes sure it sees them.

Before: *"The compound showed moderate activity against EGFR."* (hallucinated summary)
After: *"IC50 = 3.2 nM against EGFR T790M, with 47-fold selectivity over wild-type."* (actual data)

### 2. You spend less on API costs.

Lumisift selects 50% of chunks while retaining 85% of data. You send fewer tokens to the LLM and get better answers.

| Scale | Without Lumisift | With Lumisift | Savings |
|-------|-----------------|--------------|---------|
| 1 paper | ~50K tokens | ~24K tokens | 52% |
| 100 papers/day | ~$5/day (GPT-4) | ~$2.40/day | $78/month |
| 1,000 papers/month | ~$150/month | ~$72/month | $78/month |
| Enterprise (10K/month) | ~$1,500/month | ~$720/month | **$780/month** |

### 3. You know what your pipeline loses.

Even if you don't adopt Lumisift as a selection layer, the **loss taxonomy** tells you what your current retrieval system is silently discarding. That's valuable diagnostic information for any RAG team.

```bash
python information_loss_taxonomy.py
# Shows: 64% of numerical facts lost, 61% of comparisons lost, etc.
# Now you know. Now you can decide what to do about it.
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Saeedmora/Lumisift.git
cd Lumisift
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### Run the Web Interface

```bash
python app.py
# Open http://localhost:5000
```

### Run the Benchmarks

```bash
# Core diagnostics (no API key needed, runs locally)
python information_loss_taxonomy.py         # What does your retrieval lose?
python ablation_study.py                    # What drives Lumisift's results?
python information_utility_model.py         # Train the learned model

# Baseline comparisons (no API key needed)
python numerical_retention_benchmark.py     # Lumisift vs embedding retrieval
python baseline_comparison.py              # BM25 / ColBERT / Embedding
python cross_encoder_benchmark.py          # Cross-encoder reranker
python drug_discovery_usecase.py            # 3 pharma scenarios
python hybrid_benchmark.py                 # Hybrid alpha sweep

# Corpus generation
python pubmed_benchmark.py                 # Fetch 1,077 PubMed articles

# Quality evaluation (requires GEMINI_API_KEY in .env)
python downstream_eval.py                  # AI-judged answer quality
python pubmedqa_benchmark.py               # PubMedQA comprehension test

# Reproducibility
python export_reproducibility_kit.py       # Export verifiable dataset
```

### Configuration

| Variable | Required | What it does |
|----------|---------|-------------|
| `GEMINI_API_KEY` | Optional | Enables AI-judged quality evaluation |
| `HF_TOKEN` | Optional | Faster model downloads from HuggingFace |
| `MODEL_PATH` | Optional | Custom path to GGUF model |

Without API keys, everything runs **100% locally**. No data leaves your machine.

---

## Architecture

```
+-------------------------------------------------------------------+
|                        LUMISIFT                                    |
|                                                                    |
|  Input:  Text chunks from your existing pipeline                  |
|  Output: Reranked chunks prioritizing information density          |
|                                                                    |
|  Scoring backends (auto-selected):                                 |
|    Heuristic  : regex-based density detection (~0ms/chunk)        |
|    Learned MLP: trained utility model, 93K params (~0.1ms/chunk)  |
|    TinyLlama  : local LLM scoring (~200ms/chunk)                  |
|                                                                    |
|  Selection modes:                                                  |
|    'lumisift'   : maximize data retention                         |
|    'similarity' : maximize semantic relevance                     |
|    'hybrid'     : blend both (alpha=0.3 recommended)              |
+-------------------------------------------------------------------+
```

---

## Limitations (Read This)

| What's limited | Why it matters | What we're doing |
|---------------|---------------|-----------------|
| **One mechanism drives results** | Specificity alone = 90%. Other 7 axes contribute nothing to data retention. | Honest: you're adopting a data density detector, not multi-axis intelligence. |
| **Small IC50 sample** | "100% IC50 retention" is n=24. Could be dataset-specific. | We report sample sizes so you can judge. |
| **Comprehension drops** | PubMedQA: 46.7% vs 93.3% full text. Data chunks aren't explanation chunks. | Use hybrid mode (alpha=0.3) for mixed queries. |
| **Methods text gets deprioritized** | Lumisift reduces methodology retention by -3pp. | Design trade-off: data paragraphs win over procedure paragraphs. |
| **Biomedical focus** | Trained on PubMed abstracts. Legal/financial text needs retraining. | Learned model can be retrained on any domain. |
| **AI-judged quality** | Gemini evaluator has biases. Not human-validated yet. | Human evaluation is next on the roadmap. |

---

## What's Next

| Status | Milestone | Result |
|--------|-----------|--------|
| ✅ | Information loss taxonomy | 6 types characterized, 1,070 articles |
| ✅ | Ablation study | Specificity = the mechanism, everything else = noise |
| ✅ | Learned utility model | 87% retention, no regex, 368 KB |
| ✅ | 6 baseline comparisons | +42pp over best traditional method |
| ✅ | Drug discovery validation | 84% vs 15% critical data retention |
| ✅ | Reproducibility kit | 200 articles, full methodology |
| 🔜 | Multi-objective learning | Give each axis its own supervised target |
| 🔜 | LangChain / LlamaIndex plugin | Drop-in reranker for existing pipelines |
| 🔜 | Cross-domain generalization | Legal, financial, clinical text |
| 🔜 | Human evaluation | Expert ratings alongside AI judge |

---

## Project Structure

```
Lumisift/
|
|-- Core
|   |-- core/pipeline.py                # Main orchestrator (select_context API)
|   |-- core/axes_evaluator.py          # Scoring engine (heuristic + learned)
|   |-- core/embeddings.py              # MiniLM sentence embeddings
|   |-- core/atom.py                    # Data model
|   |-- app.py                          # Web interface (Flask)
|
|-- Diagnostics
|   |-- information_loss_taxonomy.py    # What does retrieval lose?
|   |-- ablation_study.py              # What component drives results?
|   |-- information_utility_model.py   # Train learned replacement
|
|-- Benchmarks
|   |-- numerical_retention_benchmark.py
|   |-- baseline_comparison.py         # BM25 / ColBERT / Embedding
|   |-- cross_encoder_benchmark.py     # Cross-encoder reranker
|   |-- drug_discovery_usecase.py      # 3 pharma scenarios
|   |-- hybrid_benchmark.py           # Alpha sweep
|   |-- pubmed_benchmark.py           # 1,077-article corpus
|   |-- pubmedqa_benchmark.py         # Comprehension test
|   |-- downstream_eval.py            # AI-judged quality
|   |-- export_reproducibility_kit.py  # Verifiable dataset
|
|-- Data
|   |-- benchmark_data/                # Generated results (JSON)
|   |-- models/                        # Trained models (.pt)
```

---

## Contributing

The highest-impact contributions right now:

1. **Run the loss taxonomy on non-biomedical text** -- Legal, financial, or clinical trial data. We want to know if the patterns hold across domains.
2. **Build a LangChain/LlamaIndex integration** -- Make Lumisift a drop-in reranker.
3. **Human evaluation** -- If you're a domain expert, your ratings on selected vs full-text quality would significantly strengthen the evidence.
4. **Improve the utility model** -- Better training signals, more diverse data, cross-domain transfer.

```bash
git clone https://github.com/Saeedmora/Lumisift.git
git checkout -b feature/your-improvement
python information_loss_taxonomy.py    # Make sure results hold
# open a PR
```

---

## License

**AGPL-3.0** -- Free to use, modify, and distribute. Source code must be shared if you deploy a modified version (including SaaS). Attribution required.

**Commercial licensing** available for proprietary use.

**Contact:** [Saeed Moradtalab](https://www.linkedin.com/in/ben-moradtalab-9442a41a6)

---

<p align="center">
  <strong>Lumisift</strong><br>
  Your retrieval system loses 64% of scientific data. Lumisift reduces that to 13%.<br>
  One mechanism. 1,077 articles. 6 baselines beaten. Every result reproducible.<br>
  <sub>Copyright 2026 Saeed Moradtalab</sub>
</p>
