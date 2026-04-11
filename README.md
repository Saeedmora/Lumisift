<p align="center">
  <strong>Lumisift</strong><br>
  <em>Measuring and Reducing Information Loss in Retrieval-Augmented Generation</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0"></a>
  <a href="#the-evidence"><img src="https://img.shields.io/badge/benchmark-1077_articles_|_6_baselines-orange?style=flat-square" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-not_required-brightgreen?style=flat-square" alt="No GPU"></a>
</p>

<p align="center">
  <sub>Created by <a href="https://www.linkedin.com/in/ben-moradtalab-9442a41a6">Saeed Moradtalab</a></sub>
</p>

---

## The Problem

Every RAG system selects context by one criterion: **semantic similarity**. That works when the answer lives in the most similar-looking paragraph. In scientific literature, it doesn't.

The paragraph that *looks* relevant -- *"We investigated EGFR inhibitors for NSCLC treatment..."* -- is background. The paragraph with the actual drug potency -- *"IC50 = 3.2 nM, 47-fold selectivity"* -- gets discarded because it doesn't resemble the query.

We measured this systematically. The result:

> **Embedding retrieval loses 64% of numerical facts, 61% of comparative claims, and 59% of causal statements.**

This isn't a retrieval problem. It's a **selection** problem. The retrieval finds the right document -- but the selection step picks the wrong paragraphs from within it.

Lumisift measures this information loss and provides a mechanism to reduce it.

---

## What Lumisift Does

Lumisift is two things:

**1. A diagnostic framework** that characterizes *what* retrieval loses -- not just *how much*. We measure loss across 6 information types (numerical, entities, causal, uncertainty, methodological, comparative) and show which types are systematically vulnerable.

**2. A selection layer** that reduces information loss by prioritizing chunks with high information density. The core mechanism is simple: detect quantitative and comparative data in a chunk, boost its selection priority by up to 1.8x. No new architecture -- just a different selection criterion.

```
Standard RAG:   text -> embed -> similarity rank -> select top-k -> LLM
Lumisift:        text -> embed -> information density score -> select top-k -> LLM
                                       |
                          "Does this chunk contain data worth preserving?"
```

### The Numbers

| | Standard RAG | Lumisift |
|-|-------------|----------|
| Numerical fact retention | 36% | **85%** |
| Comparative claim retention | 39% | **87%** |
| Token reduction | -- | **49%** |
| IC50/EC50 retention | 27% | **100%** (n=24) |
| Runs locally, no GPU | Depends | **Yes** |
| Regex-free (learned model) | N/A | **86.6%** (replaces heuristic) |

---

## The Corpus

All results in this document are based on a single, reproducible benchmark corpus:

**1,077 PubMed articles** across **10 biomedical domains**, fetched via the NCBI E-utilities API with fixed search queries.

| Domain | Articles | Focus |
|--------|---------|-------|
| Protein engineering | ~120 | Directed evolution, rational design |
| Drug discovery | ~120 | Hit-to-lead, SAR, in vivo efficacy |
| Protein extraction | ~100 | Purification, chromatography, yields |
| Enzyme optimization | ~110 | Activity, stability, enantioselectivity |
| mRNA delivery | ~100 | LNP formulation, transfection, expression |
| Antibody engineering | ~110 | Affinity maturation, humanization |
| CRISPR gene editing | ~100 | Knock-in/out efficiency, off-target |
| Biocatalysis | ~100 | Industrial enzymes, process optimization |
| Pharmacokinetics | ~110 | ADME, bioavailability, half-life |
| Vaccine development | ~107 | Immunogenicity, adjuvants, efficacy |

After filtering (abstract > 50 words), **1,070 articles** are used in all benchmarks. From these: **6,463 text chunks**, **2,722 numerical facts** (in 584 articles with quantitative data), and **4,400 labeled training samples**.

Every benchmark reads from the same `benchmark_data/pubmed_articles.json`. Regenerate with: `python pubmed_benchmark.py`

---

## The Core Insight

Every retrieval method we tested -- BM25, ColBERT, embedding similarity, cross-encoder reranking -- produces nearly identical results on information retention. They all lose 56-64% of quantitative data:

| Method | Architecture | Numerical Retention |
|--------|-------------|-------------------|
| Embedding (MiniLM) | Dense bi-encoder | 38.2% |
| BM25 (Okapi) | Sparse keyword | 41.8% |
| ColBERT | Late interaction | 43.6% |
| Cross-Encoder (ms-marco) | Joint scoring | 44.2% |
| **Lumisift (heuristic)** | **Information density** | **82.8%** |
| **Lumisift (learned)** | **Utility model** | **86.6%** |

The gap between the best traditional method (cross-encoder, 44.2%) and Lumisift (86.6%) is **+42.4 percentage points**. This gap exists because traditional methods optimize for *query-document relevance*, not for *information density*. They answer "does this text match?" when the real question is "does this text contain data worth preserving?"

---

## The Evidence

Every result is reproducible. Every script is included. We show our failures alongside our strengths.

---

### 1. Information Loss Taxonomy

**The scientific core.** We characterized *what* embedding retrieval systematically loses across 1,070 articles:

| Information Type | Items Tested | Embedding Retains | Lumisift Retains | Loss Severity | Recoverable? |
|-----------------|-------------|-------------------|-----------------|---------------|-------------|
| **Numerical Facts** | 1,109 | 36.0% | **85.3%** | SEVERE | YES (+49pp) |
| **Comparative Claims** | 136 | 38.6% | **87.0%** | SEVERE | YES (+48pp) |
| **Causal Statements** | 920 | 40.7% | **50.1%** | SEVERE | PARTIAL (+9pp) |
| **Uncertainty Markers** | 639 | 45.9% | 48.5% | SEVERE | NO (+3pp) |
| **Methodological Details** | 419 | 59.5% | 56.5% | MODERATE | NO (-3pp) |
| **Named Entities** | 2,992 | 65.2% | 70.6% | MODERATE | PARTIAL (+5pp) |

**What this reveals:**

1. **Numerical facts and comparative claims** are the most vulnerable information types in embedding retrieval. Lumisift recovers nearly all of this loss (+49pp, +48pp).
2. **Causal statements** are partially recoverable (+9pp). The specificity mechanism helps because causal text often co-occurs with numerical evidence, but dedicated causal detection would do better.
3. **Uncertainty markers and methodological details are NOT recoverable** by Lumisift. The specificity boost doesn't help here -- and actually hurts method retention (-3pp) because it deprioritizes procedural text.
4. **Named entities** are relatively well-preserved by all methods (65-71%). Entities appear across most paragraphs, so the selection method matters less.

This taxonomy is the project's primary scientific contribution. It shifts the question from *"which method is best?"* to *"what does each method lose, and is that loss recoverable?"*

`python information_loss_taxonomy.py`

---

### 2. Ablation Study -- What Actually Works

We systematically removed each scoring component to determine what drives the results:

| Configuration | Retention | Delta | Verdict |
|--------------|-----------|-------|---------|
| **Only specificity** | **90.0%** | **+7.3pp** | **Outperforms full system** |
| Full system (8 axes) | 82.7% | baseline | +/-2.7pp CI |
| Without relevance | 88.6% | +5.9pp | Removing it *helps* |
| Without specificity | 48.4% | -34.3pp | System collapses |
| Without trust | 82.8% | +0.1pp | No effect |
| Without risk | 83.9% | +1.2pp | No effect |
| Without causality/temporal/ontology/visibility | 82.7% | +0.0pp | No effect |

**The honest conclusion:** One mechanism does the work -- the specificity boost (1.0-1.8x multiplier on data-dense chunks). The other 7 axes contribute nothing to numerical retention. Relevance actually *hurts* because it favors descriptive text over data paragraphs.

This doesn't mean the other axes are useless. It means they need **separate supervised targets** (trust needs citation labels, causality needs entailment labels). Currently they're noise in a single-objective system.

`python ablation_study.py`

---

### 3. Learned Utility Model -- Beyond Heuristics

The heuristic specificity detector uses regex. That's fragile and domain-specific. We trained a neural replacement:

**Model:** 384 → 192 → 96 → 1 MLP (93K params, 368 KB). Trained on 6,463 chunks from 1,070 articles. Training signal: actual information density (numerical + entity + causal content per chunk), not heuristic labels.

| Method | Numerical Retention | Regex-Free? |
|--------|-------------------|-------------|
| Embedding similarity | 40.6% | Yes |
| Heuristic (regex specificity) | 81.3% | No |
| **Learned utility model** | **86.6%** | **Yes** |

**The learned model beats the heuristic by +5.4pp** -- and it doesn't depend on regex patterns. It learns to detect information density directly from embeddings. This is the path to domain generalization: the model can be retrained on legal, financial, or clinical text without rewriting pattern rules.

`python information_utility_model.py`

---

### 4. Baseline Comparison -- 6 Methods Head-to-Head

| Method | Retention | vs BM25 | 95% CI |
|--------|-----------|---------|--------|
| **Utility Model (learned)** | **86.6%** | **+44.8pp** | -- |
| **Lumisift (heuristic)** | **82.8%** | **+41.0pp** | +/-2.7pp |
| **Hybrid (alpha=0.3)** | **75.5%** | **+33.7pp** | +/-3.1pp |
| Cross-Encoder (ms-marco) | 44.2% | +2.4pp | +/-3.4pp |
| ColBERT (late interaction) | 43.6% | +1.8pp | +/-3.4pp |
| BM25 (Okapi) | 41.8% | baseline | +/-3.4pp |
| Embedding (MiniLM cosine) | 38.2% | -3.6pp | +/-3.4pp |

`python baseline_comparison.py` | `python cross_encoder_benchmark.py`

---

### 5. Drug Discovery -- Downstream Impact

| Scenario | Standard RAG | Lumisift |
|----------|-------------|----------|
| EGFR inhibitor (IC50, selectivity) | 44% | **67%** |
| Lipase evolution (kcat/Km, E-value) | **0%** | **86%** |
| mRNA vaccine (fold-change, ED50, PDI) | **0%** | **100%** |

The lipase case: embedding retrieval kept **zero** kinetic parameters. It selected background text instead of the results section. A researcher would never know the measurements existed.

`python drug_discovery_usecase.py`

---

### 6. Where Lumisift Fails

**PubMedQA comprehension:** 46.7% accuracy vs 93.3% for full text and embedding. Lumisift selects data paragraphs, not explanation paragraphs. For yes/no comprehension, that's the wrong choice.

**Methodological details:** Lumisift *reduces* retention of procedural text by -3pp. The specificity boost deprioritizes methods sections.

**Uncertainty markers:** Not recoverable. Hedging language ("may", "suggests", "preliminary") doesn't correlate with data density.

**Hybrid mode** (alpha=0.3) partially addresses the comprehension weakness: 72.4% numerical retention with improved comprehension signal.

`python pubmedqa_benchmark.py` | `python hybrid_benchmark.py`

---

### 7. Reproducibility Kit

200 articles exported with pre-computed scores, 818 numerical facts, full methodology documentation, and selected chunk indices. Anyone can verify the results.

`python export_reproducibility_kit.py` → `benchmark_data/reproducibility_kit.json` (1 MB)

---

## Who Should Use This

Lumisift is designed for teams where **a missing number changes the answer.**

| Domain | What Gets Lost Without Lumisift |
|--------|-------------------------------|
| **Pharmaceutical R&D** | IC50 values, selectivity ratios, dosing data |
| **Protein Engineering** | kcat/Km, fold-changes, mutation effects |
| **Clinical Research** | p-values, confidence intervals, hazard ratios |
| **Regulatory Affairs** | Exact concentration limits, quantitative thresholds |

### When NOT to Use Lumisift

- **General Q&A** -- if you don't need specific numbers, standard RAG is fine
- **Comprehension tasks** -- for yes/no questions, use embedding or hybrid mode
- **Non-scientific domains** (without retraining) -- the learned model is trained on biomedical text

### Cost Impact

| Scenario | Without | With Lumisift |
|----------|---------|---------------|
| 100-page paper | ~50K tokens | ~24K tokens (52% savings) |
| Monthly (1,000 papers, GPT-4) | ~$150 | ~$72 |
| Numerical accuracy | 36% available | **85% available** |

---

## Architecture

```
+-------------------------------------------------------------------+
|                     LUMISIFT PIPELINE                              |
|                                                                    |
|  Scoring Backends (choose one):                                    |
|    1. Heuristic   : regex specificity detection (~0ms)            |
|    2. Learned MLP : trained utility model (93K params, ~0.1ms)    |
|    3. TinyLlama   : GGUF Q4 local LLM (~200ms)                   |
|                                                                    |
|  Selection Modes:                                                  |
|    'lumisift'   : pure information density (best for data)        |
|    'similarity' : pure embedding (best for comprehension)         |
|    'hybrid'     : alpha-blend (alpha=0.3 recommended)             |
|                                                                    |
|  Diagnostic Tools:                                                 |
|    Loss taxonomy    : what does retrieval lose?                    |
|    Ablation study   : which component drives results?             |
|    Baseline suite   : how does it compare to 6 methods?           |
+-------------------------------------------------------------------+
```

---

## Getting Started

```bash
git clone https://github.com/Saeedmora/Lumisift.git
cd Lumisift
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .

python app.py                 # Web interface at http://localhost:5000
```

### Python API

```python
from core.pipeline import LogicalRoomsPipeline

pipe = LogicalRoomsPipeline(verbose=True)

# Select context (hybrid mode recommended for general use)
result = pipe.select_context(
    chunks, query="EGFR inhibitor IC50",
    mode="hybrid", alpha=0.3, top_k=5
)
print(f"Tokens saved: {result.compression_ratio:.0%}")
```

### Run All Benchmarks

```bash
# Diagnostic (no API key needed)
python information_loss_taxonomy.py         # What does retrieval lose? (core result)
python ablation_study.py                    # Which component matters?
python information_utility_model.py         # Train learned replacement

# Baselines (no API key needed)
python numerical_retention_benchmark.py     # Lumisift vs embedding
python baseline_comparison.py              # BM25 / ColBERT / Embedding
python cross_encoder_benchmark.py          # Cross-encoder reranker

# Use cases (no API key needed)
python drug_discovery_usecase.py            # 3 pharma scenarios
python hybrid_benchmark.py                 # Alpha sweep
python pubmed_benchmark.py                 # 1,077-article corpus

# Quality evaluation (requires GEMINI_API_KEY)
python downstream_eval.py                  # AI-judged answer quality
python pubmedqa_benchmark.py               # PubMedQA comprehension test

# Reproducibility
python export_reproducibility_kit.py       # Export verifiable dataset
```

---

## Known Limitations

| Limitation | Impact | Status |
|-----------|--------|--------|
| **Single dominant mechanism** | Ablation shows specificity alone outperforms the full system. Other axes need separate supervision to contribute. | Documented. Learned model replaces heuristics. |
| **IC50 sample size** | "100% IC50 retention" is based on n=24. Could overfit. | Sample size reported alongside claim. |
| **Comprehension weakness** | 46.7% on PubMedQA vs 93.3% for full text | Hybrid mode (alpha=0.3) partially addresses |
| **Methodology loss** | Lumisift *reduces* retention of procedural text by -3pp | Fundamental: density prioritization deprioritizes methods |
| **Uncertainty not recoverable** | Hedging language doesn't correlate with data density | Needs dedicated uncertainty model |
| **Domain specificity** | Trained on biomedical text. Legal/financial needs retraining. | Learned model enables retraining without regex |
| **AI judge bias** | Gemini 3 Flash evaluator has its own preferences | Human evaluation planned |

---

## Roadmap

| Status | Milestone | Key Result |
|--------|-----------|------------|
| Done | Information loss taxonomy | 6 information types characterized across 4 methods |
| Done | Ablation study | Specificity alone = 90.0%, 7 other axes = negligible |
| Done | Learned utility model | 86.6% retention, beats heuristic by +5.4pp, no regex |
| Done | 6-method baseline comparison | +42pp over cross-encoder, +41pp over BM25 |
| Done | Drug discovery validation | 84% vs 15% critical data retention |
| Done | Hybrid mode | Alpha=0.3 balances data + comprehension |
| Done | Reproducibility kit | 200 articles, full methodology, verifiable |
| Next | Multi-objective learning | Give each axis its own supervised target |
| Next | Cross-domain generalization | Legal, financial, clinical text |
| Next | LangChain / LlamaIndex plugin | Drop-in reranker |
| Next | Human evaluation | Expert ratings alongside AI judge |

---

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Optional | AI-judged quality evaluation |
| `HF_TOKEN` | Optional | Faster HuggingFace downloads |
| `MODEL_PATH` | Optional | Path to local GGUF model |

Without any API keys, everything runs **100% locally**. No data leaves your machine.

---

## Project Structure

```
Lumisift/
|-- app.py                              # Flask web server + API
|-- information_loss_taxonomy.py        # Core: what does retrieval lose?
|-- information_utility_model.py        # Learned utility scorer (replaces heuristics)
|-- ablation_study.py                   # Which component drives results?
|-- numerical_retention_benchmark.py    # Lumisift vs embedding
|-- baseline_comparison.py              # BM25 / ColBERT / Embedding
|-- cross_encoder_benchmark.py          # Cross-encoder reranker baseline
|-- drug_discovery_usecase.py           # 3 pharma scenarios
|-- hybrid_benchmark.py                 # Alpha sweep
|-- learned_scoring.py                  # MLP axis predictor (early version)
|-- pubmed_benchmark.py                 # 1,077-article corpus
|-- pubmedqa_benchmark.py              # PubMedQA comprehension test
|-- downstream_eval.py                  # AI-judged answer quality
|-- export_reproducibility_kit.py       # Verifiable dataset export
|-- core/
|   |-- axes_evaluator.py              # Scoring engine (heuristic + learned)
|   |-- pipeline.py                    # Orchestrator (3 selection modes)
|   |-- atom.py                        # Data model
|   |-- embeddings.py                  # Sentence-transformer embeddings
|   |-- surface.py                     # Clustering
|   |-- finetuning.py                  # Calibration + training export
|   |-- self_optimization.py           # Room splitting
|-- benchmark_data/                    # Generated results (JSON)
|-- models/                            # Trained models (.pt)
```

---

## Contributing

The highest-impact contributions:

1. **Cross-domain benchmarks** -- Run the loss taxonomy on legal, financial, or clinical text
2. **Better utility signals** -- Improve the training signal for the learned model
3. **RAG integrations** -- Lumisift as a LangChain/LlamaIndex/Haystack reranker
4. **Human evaluation** -- Domain expert ratings on selected vs full-text quality

```bash
git clone https://github.com/Saeedmora/Lumisift.git
git checkout -b feature/your-improvement
python information_loss_taxonomy.py    # Verify results
# open a PR
```

---

## License

**AGPL-3.0** -- Free to use, modify, and distribute. Source must be shared for deployed modifications (including SaaS). Attribution required.

**Commercial licensing** available for proprietary use cases.

**Contact:** [Saeed Moradtalab](https://www.linkedin.com/in/ben-moradtalab-9442a41a6)

---

<p align="center">
  <strong>Lumisift</strong><br>
  Embedding retrieval loses 64% of numerical facts. We measure that loss and reduce it to 13%.<br>
  Validated on 1,077 articles against 6 baselines. Every benchmark reproducible.<br>
  <sub>Copyright 2026 Saeed Moradtalab</sub>
</p>
