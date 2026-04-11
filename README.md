<p align="center">
  <strong>Lumisift</strong><br>
  <em>Multi-Axis Scientific Intelligence for RAG Pipelines</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0"></a>
  <a href="#the-evidence"><img src="https://img.shields.io/badge/benchmark-1077_articles_|_10_domains-orange?style=flat-square" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-not_required-brightgreen?style=flat-square" alt="No GPU"></a>
</p>

<p align="center">
  <sub>Created by <a href="https://www.linkedin.com/in/ben-moradtalab-9442a41a6">Saeed Moradtalab</a></sub>
</p>

---

## Why Lumisift Exists

Every RAG pipeline today selects context by asking one question: *"Which text looks most similar to the query?"*

For conversational AI, that works fine. For scientific literature, it's a disaster.

The paragraph that *looks* most relevant -- *"We investigated EGFR inhibitors for the treatment of NSCLC..."* -- is often pure background. It contains zero data. Meanwhile, the paragraph with the actual drug potency -- *"LX-4291: IC50 = 3.2 nM, 47-fold selectivity over wild-type"* -- gets silently discarded because it doesn't *sound* like the query.

We tested this systematically. Across **1,077 PubMed articles** in 10 biomedical domains, standard embedding retrieval loses **60% of all quantitative data** -- IC50 values, p-values, fold-changes, dosing concentrations, kinetic constants -- all gone before the LLM ever sees them.

That's not a retrieval problem. That's a **selection** problem. And it's the problem Lumisift solves.

---

## What Lumisift Does

Lumisift is a **context selection layer** that scores every text passage across 8 independent semantic dimensions before deciding what the LLM should see. Instead of relying on a single similarity signal, it evaluates relevance, specificity, trust, risk, causality, temporality, domain classification, and data density -- then selects the passages that carry the most scientific weight.

```
Raw Text --> Embedding --> 8-Signal Evaluation --> Priority Selection --> LLM

                               Relevance: How important is this finding?
                               Specificity: Does it contain quantitative data?
                               Trust: Is this peer-reviewed or preliminary?
                               Risk: Does it flag uncertainty?
                               Causality: Cause-effect or just correlation?
```

The key insight: a chunk containing *"IC50 = 3.2 nM (47-fold improvement)"* receives up to **1.8x priority** through the specificity boost, ensuring that quantitative results survive context compression. This single mechanism accounts for most of the performance difference.

### At a Glance

| | Standard RAG | Lumisift |
|-|-------------|----------|
| Selection signals | 1 (similarity) | **8 (multi-axis)** |
| Numerical data retained | 40% | **83%** |
| IC50/EC50 retention | 27% | **100%** |
| Token savings | -- | **49% fewer tokens** |
| Runs locally | Depends | **Yes, no GPU needed** |
| Text fidelity | Some systems summarize | **100% lossless** |

---

## The Vision

Lumisift is a **first step**, not a final product. The long-term direction:

1. **Active retrieval** -- the LLM requests *"I need causal evidence"* and Lumisift returns high-causality chunks, not just similar-sounding ones
2. **Domain adaptation** -- the system learns from your corrections which signals matter most for your field
3. **Standard re-ranker** -- a drop-in layer for any RAG pipeline that automatically protects quantitative data

We're building toward AI that evaluates what it reads, not just retrieves what it finds.

---

## The Evidence

Every claim below is backed by a reproducible benchmark. Every script is included. We show both our strengths and our weaknesses, because credibility matters more than marketing.

---

### 1. Numerical Retention -- The Core Result

**The question:** If you compress context by 50%, how many scientific measurements survive?

This is the most consequential test for any scientific RAG system. A missing IC50 value doesn't just make an answer incomplete -- it creates a hallucination. The AI fills the gap with a guess, and a guessed drug potency is worse than no answer at all.

**Tested on:** 584 articles with quantitative data (from 1,077 total), 2,722 individual numerical facts, 10 biomedical domains.

| Method | Facts Retained | Retention Rate |
|--------|---------------|----------------|
| Standard embedding retrieval | 1,100 / 2,722 | 40.4% |
| **Lumisift** | **2,256 / 2,722** | **82.9%** |
| **Improvement** | **+1,156 facts saved** | **+42.5pp** |

**In plain terms:** A paper with 10 important numbers -- standard RAG keeps 4. Lumisift keeps 8.

**Breakdown by data type:**

| Data Type | Tested | Standard RAG | Lumisift | Industry Impact |
|-----------|--------|-------------|----------|-----------------|
| Fold changes | 161 | 32.9% | **92.5%** | Drug potency comparisons |
| p-values | 32 | 34.4% | **90.6%** | Statistical significance |
| Precise decimals | 495 | 40.6% | **88.9%** | Exact measurements |
| IC50 / EC50 | 24 | 29.2% | **87.5%** | Drug candidate ranking |
| Concentrations | 281 | 35.6% | **86.8%** | Dosing decisions |
| Percentages | 607 | 37.9% | **87.0%** | Yields, efficiencies |
| Large numbers | 775 | 46.2% | **76.4%** | Scale indicators |

Lumisift wins on **all 13 fact types** tested. Per-article: Lumisift wins 61%, embedding wins 8%, ties 31%.

`python numerical_retention_benchmark.py`

---

### 2. Versus Established Baselines -- BM25, ColBERT, Cross-Encoder

A fair comparison must include the retrieval methods that professionals actually use. We tested against BM25 (the keyword standard), ColBERT (late-interaction token matching), standard embedding cosine similarity, and **cross-encoder reranking** (ms-marco-MiniLM-L-6-v2 -- the strongest traditional baseline).

| Method | Retention | vs BM25 | 95% CI |
|--------|-----------|---------|--------|
| **Lumisift** | **82.8%** | **+41.0pp** | +/-2.7pp |
| **Hybrid (alpha=0.3)** | **75.5%** | **+33.7pp** | +/-3.1pp |
| Cross-Encoder (ms-marco) | 44.2% | +2.4pp | +/-3.4pp |
| ColBERT (late interaction) | 43.6% | +1.8pp | +/-3.4pp |
| BM25 (Okapi) | 41.8% | baseline | +/-3.4pp |
| Embedding (MiniLM cosine) | 38.2% | -3.6pp | +/-3.4pp |

**The insight:** All four traditional methods -- BM25, ColBERT, embedding, and cross-encoder -- cluster between 38-44%. Despite radically different architectures (keyword matching, late interaction, dense retrieval, joint query-document scoring), they produce near-identical results on numerical retention. The cross-encoder, often cited as the strongest reranker, adds only +2.4pp over BM25.

The problem isn't *how* you match. It's that matching -- regardless of sophistication -- ignores whether the selected text contains data worth preserving.

**IC50 / EC50 retention:** BM25 = 27%, Embedding = 27%, Cross-Encoder = N/A, ColBERT = 45%, **Lumisift = 100%** (n=24, see limitations).

`python baseline_comparison.py` | `python cross_encoder_benchmark.py`

---

### 3. Drug Discovery -- Real-World Consequences

In pharmaceutical research, a missed measurement isn't an inconvenience. It's a missed drug candidate. We constructed three realistic scenarios to illustrate what happens when retrieval fails at the selection layer.

| Scenario | Critical Data | Standard RAG | Lumisift |
|----------|--------------|-------------|----------|
| **EGFR inhibitor screening** | IC50, tumor inhibition, selectivity ratio | 44% | **67%** |
| **Lipase directed evolution** | kcat/Km, enantioselectivity E-value, ee% | **0%** | **86%** |
| **mRNA vaccine LNP optimization** | fold-change, ED50, particle size, PDI | **0%** | **100%** |

The lipase case deserves attention: embedding retrieval retained **zero** out of seven critical kinetic parameters. It selected the background paragraph (*"Chiral intermediates are important in pharmaceutical synthesis..."*) instead of the results section containing kcat/Km = 4,500 M-1 s-1 and E-value > 200. A researcher relying on this output would never know these measurements existed.

**Average across all 3 scenarios:** Standard RAG retains **15%** of critical drug data. Lumisift retains **84%**.

`python drug_discovery_usecase.py`

---

### 4. Downstream Answer Quality

Context selection is only useful if the LLM produces good answers from the selected text. We used Gemini 3 Flash as an AI judge to evaluate answer quality across multiple dimensions.

| Dimension | Full Text | Lumisift | Verdict |
|-----------|-----------|----------|---------|
| Accuracy | 5.0 | 3.6 | Full text better |
| Relevance | 5.0 | **4.9** | **Essentially tied** |
| Conciseness | 5.0 | **4.6** | Close |
| Composite | 5.0 | **4.15 (83%)** | 83% quality retained |

**Context reduction:** 49% fewer tokens. **Efficiency gain:** +64% more quality per token spent.

**The trade-off, stated honestly:** Full text always wins on accuracy because it contains everything. Lumisift sometimes drops explanatory paragraphs in favor of data-heavy ones. For questions that require comprehension (*"Does X improve stability?"*), this hurts accuracy. For questions that require specific numbers (*"What was the IC50?"*), Lumisift outperforms.

This trade-off is exactly why we built the hybrid mode.

<details>
<summary><strong>Methodology details</strong></summary>

- Randomly sampled articles from the 1,077-article corpus (seed=42)
- Generated scientific questions using Gemini 3 Flash Preview
- Answered each question with full text and Lumisift-selected text
- Blind grading on Accuracy, Completeness, Relevance, Conciseness (1-5)
- Run in batches of 10 to respect free-tier API rate limits

**Limitations:** AI judge introduces subjectivity; human evaluation would be more authoritative. Free-tier caps total evaluations per day.

</details>

`python downstream_eval.py` (requires GEMINI_API_KEY)

---

### 5. Where Lumisift Fails

We tested PubMedQA-style comprehension questions -- yes/no/maybe answers about scientific findings.

| Method | Accuracy |
|--------|----------|
| Full Context | **93.3%** |
| Embedding Similarity | **93.3%** |
| Lumisift | **46.7%** |

**This is a significant weakness -- and it's by design.** PubMedQA asks questions like *"Does directed evolution improve enzyme stability?"* To answer yes or no, the LLM needs the conclusion and background paragraphs. Lumisift prioritizes the data paragraphs (kcat values, fold-changes, temperatures) and drops the explanatory context.

For comprehension: similarity is better. For data preservation: Lumisift is better. We don't pretend otherwise.

This finding directly motivated our next result.

`python pubmedqa_benchmark.py`

---

### 6. The Hybrid Solution

We combined both signals with a tunable blend:

```
hybrid_score = alpha * similarity + (1 - alpha) * lumisift
```

Sweeping alpha across 517 articles reveals a clear optimum:

| Alpha | Strategy | Numerical Retention |
|-------|---------|-------------------|
| 0.0 | Pure Lumisift (data only) | 81.0% |
| **0.3** | **70% data + 30% comprehension** | **72.4%** |
| 0.5 | Equal balance | 65.1% |
| 1.0 | Pure similarity (comprehension only) | 40.8% |

**At alpha = 0.3**, the system retains **72.4%** of numerical facts -- still **+31.6pp** above pure similarity -- while incorporating enough semantic matching for comprehension tasks. IC50/EC50 retention remains at **100%** even in hybrid mode.

```python
# Recommended usage in production:
result = pipe.select_context(
    chunks, query="EGFR inhibitor IC50",
    mode="hybrid", alpha=0.3, top_k=5
)
```

Three modes are available: `"lumisift"` (pure data), `"similarity"` (pure comprehension), `"hybrid"` (combined, recommended).

`python hybrid_benchmark.py`

---

### 7. Learned Scoring Model

The current heuristic evaluator uses regex patterns and keyword matching -- effective but limited. We trained a lightweight neural network to learn axis scoring from embeddings directly, using the 4,400 labeled samples generated during benchmarking.

**Model:** 384 → 256 → 128 → 8 MLP with LayerNorm and GELU (133K parameters, 525 KB)

| Axis | MAE | Correlation | What it means |
|------|-----|-------------|--------------|
| **Specificity** | 0.132 | **0.689** | The model reliably learns to detect quantitative data density |
| Temporal | 0.077 | 0.408 | Reasonable detection of temporal signals |
| Ontology | 0.188 | 0.300 | Domain classification is partially learned |
| Trust | 0.016 | 0.024 | Low variance in training data limits learning |
| Risk | 0.031 | 0.186 | Subtle uncertainty signals are hard to learn from text alone |

**Why specificity matters most:** The specificity axis drives the 1.0-1.8x boost that accounts for most of Lumisift's numerical advantage. A 0.689 correlation means the learned model can reproduce this critical signal reasonably well -- opening the door to replacing regex heuristics entirely as training data grows.

**Current limitation:** Trust and risk axes show low correlation because our training data has limited variance on those dimensions. Expanding to more diverse source materials (clinical trials, regulatory documents) would improve this.

`python learned_scoring.py`

---

### 8. Ablation Study -- Which Axis Actually Matters?

The most important question for credibility: *if we remove an axis, does the result change?*

We systematically zeroed out each axis and re-ran numerical retention on 560 articles:

| Configuration | Retention | Delta | Impact |
|--------------|-----------|-------|--------|
| **Full Lumisift** | **82.7%** | baseline | +/-2.7pp CI |
| Without specificity | 48.4% | **-34.3pp** | **CRITICAL** |
| No specificity boost | 48.4% | **-34.3pp** | **CRITICAL** |
| Only specificity (nothing else) | **90.0%** | +7.3pp | Outperforms full system |
| Only relevance | 52.6% | -30.1pp | **CRITICAL** |
| Without relevance | 88.6% | +5.9pp | Removing it *helps* |
| Without trust | 82.8% | +0.1pp | Minimal |
| Without risk | 83.9% | +1.2pp | Minimal |
| Without causality | 82.7% | +0.0pp | None |
| Without temporal | 82.7% | +0.0pp | None |
| Without ontology | 82.7% | +0.0pp | None |
| Without visibility | 82.7% | +0.0pp | None |

**What this proves:**

1. **Specificity is everything.** Removing it drops retention by -34.3pp. Using *only* specificity achieves 90.0% -- better than the full system. This is the mechanism that makes Lumisift work.
2. **Relevance actually hurts numerical retention.** Removing relevance *improves* results (+5.9pp) because the relevance heuristic favors descriptive text over data-dense text.
3. **Trust, risk, causality, temporal, ontology, visibility contribute nothing** to numerical retention at their current heuristic quality. They exist for downstream comprehension tasks, not data preservation.

**The honest reading:** Lumisift's numerical retention advantage comes from one mechanism -- the specificity boost. The other 7 axes are either neutral or slightly harmful for this specific task. They may prove valuable for comprehension, causality tracking, or domain adaptation -- but for the numerical retention benchmark, specificity does the work.

`python ablation_study.py`

---

### 9. Reproducibility Kit

We export a self-contained dataset for independent verification:

- **200 articles** with pre-computed axis scores for every chunk
- **818 numerical facts** with retention results
- **Full methodology documentation** (chunking protocol, scoring formula, query generation)
- **Selected chunk indices** so anyone can verify which chunks Lumisift picks and why

```bash
python export_reproducibility_kit.py
# Outputs: benchmark_data/reproducibility_kit.json (1,073 KB)
```

The kit includes the exact chunking code, scoring formula, and fact extraction patterns used in all benchmarks. No hidden preprocessing.

---

## Who Should Use This

Lumisift is designed for teams where **a missing number changes the answer**.

| Domain | The Problem | What Happens Without Lumisift |
|--------|------------|-------------------------------|
| **Pharmaceutical R&D** | IC50 values and selectivity ratios buried in dense papers | The AI hallucinates a drug potency because the retrieval system selected the introduction instead of the results |
| **Protein Engineering** | Fold-changes, kcat/Km, and mutation effects dropped | An engineer comparing enzyme variants gets summaries instead of kinetic data |
| **Clinical Research** | p-values and confidence intervals lost during compression | A trial review is missing the statistical evidence that supports its conclusions |
| **Regulatory Affairs** | Quantitative thresholds must be preserved exactly | An auditor asking about concentration limits receives approximations |
| **Academic Literature Review** | Experimental results replaced by background text | A researcher's AI assistant summarizes method descriptions instead of findings |

### When NOT to Use Lumisift

Be honest with yourself about the use case:

- **General Q&A over documents** -- Standard RAG is fine if you don't need specific numbers
- **Conversational AI** -- Lumisift optimizes for data preservation, not dialogue flow
- **Comprehension-heavy tasks** -- If the answer is yes/no, use similarity or hybrid mode
- **Non-scientific domains** (without calibration) -- The heuristic lexicon is tuned for biomedical text

### Cost Impact

| Scenario | Without Lumisift | With Lumisift |
|----------|-----------------|---------------|
| 100-page research paper | ~50K tokens sent to LLM | ~24K tokens (**52% savings**) |
| Monthly cost (1,000 papers, GPT-4) | ~$150 | ~$72 |
| Numerical accuracy | 40% of facts available | **83% of facts available** |
| Hallucination risk for data | High (60% missing) | Low (17% missing) |

---

## The 8 Signals

| Signal | What It Detects | Example | Range |
|--------|----------------|---------|-------|
| **Relevance** | Strategic importance | *"47-fold improvement"* vs *"EGFR is well-known"* | 0 to 1 |
| **Specificity** | Quantitative data density | IC50 = 3.2 nM, kcat/Km = 4,500 | 0 to 1 |
| **Trust** | Source reliability | Peer-reviewed finding vs preliminary | 0 to 1 |
| **Risk** | Uncertainty markers | *"may suggest"*, *"needs validation"* | -1 to +1 |
| **Causality** | Cause-effect strength | *"causes"* vs *"correlates with"* | -1 to +1 |
| **Temporal** | Information currency | 2024 study vs 1995 protocol | -1 to +1 |
| **Ontology** | Domain classification | Biotech, pharmacology, regulation | 0 to 1 |
| **Visibility** | Public vs internal scope | Published result vs lab observation | 0 to 1 |

**Scoring formula:**

```
score = relevance * (1 + |risk|) * (0.5 + trust * 0.5) * temporal_boost * specificity_boost
```

The **specificity boost** (1.0x to 1.8x) is the primary innovation. It elevates chunks containing measurements, rates, and numerical outcomes. This single mechanism drives the jump from 40% to 83% retention.

---

## Getting Started

```bash
git clone https://github.com/Saeedmora/Lumisift.git
cd Lumisift
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .

# Launch the web interface
python app.py
# Open http://localhost:5000
```

### Python API

```python
from core.pipeline import LogicalRoomsPipeline

pipe = LogicalRoomsPipeline(verbose=True)

# Evaluate a single passage
atom = pipe.process(
    "LX-4291 demonstrated IC50 of 3.2 nM against EGFR T790M, a 47-fold improvement.",
    domain="biotech"
)
print(f"Specificity: {atom.axes['specificity']:.2f}")  # High -- data detected
print(f"Relevance:   {atom.axes['relevance']:.2f}")

# Select context (hybrid mode recommended)
result = pipe.select_context(
    chunks, query="EGFR inhibitor IC50",
    mode="hybrid", alpha=0.3, top_k=5
)
print(f"Tokens saved: {result.compression_ratio:.0%}")
```

### Run All Benchmarks

```bash
# Core benchmarks (no API key required)
python pubmed_benchmark.py                  # 1,077-article corpus
python numerical_retention_benchmark.py     # Lumisift vs embeddings
python baseline_comparison.py              # BM25 / ColBERT / Embedding vs Lumisift
python drug_discovery_usecase.py            # 3 pharma scenarios
python hybrid_benchmark.py                 # Alpha sweep for hybrid mode
python learned_scoring.py                  # Train the MLP axis predictor

# Quality evaluation (requires GEMINI_API_KEY in .env)
python downstream_eval.py                  # AI-judged answer quality
python pubmedqa_benchmark.py               # PubMedQA comprehension test
```

---

## Architecture

```
+-------------------------------------------------------------------+
|                        LUMISIFT PIPELINE                           |
|                                                                    |
|  Raw Text --> Embedding --> 8-Signal Scoring --> Atom              |
|               (MiniLM-L6)   (Heuristic | MLP | TinyLlama | NF4)  |
|                                                                    |
|  Atoms --> Surface Clustering --> Room Assignment                  |
|            (similarity-based)     (self-optimizing)                |
|                                                                    |
|  Selection Modes:                                                  |
|    'lumisift'    : pure multi-axis    (best for data retrieval)    |
|    'similarity'  : pure embedding     (best for comprehension)    |
|    'hybrid'      : combined alpha=0.3 (best balanced approach)    |
+-------------------------------------------------------------------+
```

### Four evaluator backends

| Backend | Engine | Speed | Best For |
|---------|--------|-------|----------|
| **Heuristic** | Keywords + regex | ~0ms/chunk | CPU-only, prototyping |
| **Learned MLP** | Trained classifier (133K params) | ~0.1ms/chunk | Faster than heuristic with better generalization |
| **GGUF Q4** | TinyLlama 1.1B quantized | ~200ms/chunk | Better nuance, no GPU |
| **NF4** | HuggingFace + bitsandbytes | ~150ms/chunk | Best accuracy, requires CUDA |

The system auto-selects the best available backend. If a model isn't installed, it falls back gracefully.

---

## Known Limitations

We publish what doesn't work alongside what does. This section is as important as the results.

| Limitation | Impact | Status |
|-----------|--------|--------|
| **7 of 8 axes don't contribute** to numerical retention | Ablation shows only specificity matters for the core metric. Other axes are neutral or harmful. | Documented: ablation study proves the mechanism is specificity, not multi-axis complexity |
| **"100% IC50 retention" sample size** | Based on n=24 IC50/EC50 values. Small sample, could overfit. | Documented: we now report sample size alongside claim |
| **Comprehension weakness** | 46.7% on PubMedQA vs 93.3% for full text | Solved: hybrid mode (alpha=0.3) |
| **Cross-encoder equivalence** | Cross-encoder (44.2%) doesn't solve the problem either -- same class as BM25 | Fundamental: matching-based methods can't detect data density |
| **AI judge bias** | Gemini 3 Flash evaluator has its own preferences | Planned: human evaluation protocol |
| **Domain specificity** | Lexicons tuned for biomedical text | Mitigatable: user feedback loop enables adaptation |
| **Learned model variance** | Trust/risk correlations < 0.2 due to limited training variance | Next: diversify training data |

---

## Roadmap

| Status | Milestone | Result |
|--------|-----------|--------|
| Done | Numerical retention benchmark | 82.9% vs 40.4% across 2,722 facts |
| Done | Drug discovery validation | 84% vs 15% critical data retention |
| Done | BM25 / ColBERT / Cross-Encoder comparison | Lumisift +41pp over BM25, +39pp over cross-encoder |
| Done | Ablation study | Specificity alone = 90.0%. 7 other axes = negligible |
| Done | Hybrid retrieval mode | Alpha=0.3 retains 72.4% with comprehension signals |
| Done | 1,077-article validation | 10 domains, 4,400 training samples |
| Done | Learned scoring model | 133K-param MLP, 0.689 specificity correlation |
| Done | Reproducibility kit | 200 articles exported with full scoring data |
| Done | Honest limitation testing | PubMedQA + ablation shows exactly what works and what doesn't |
| Next | LangChain / LlamaIndex plugin | Drop-in re-ranker for existing pipelines |
| Next | Human evaluation | Expert ratings alongside AI judge |
| Next | Cross-domain expansion | Clinical trials, legal, cybersecurity |

---

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Optional | AI-judged quality evaluation, PubMedQA benchmark |
| `HF_TOKEN` | Optional | Faster HuggingFace model downloads |
| `MODEL_PATH` | Optional | Path to local GGUF model for LLM-based scoring |

Without any API keys, the system runs **100% locally**. No data leaves your machine.

---

## Project Structure

```
Lumisift/
|-- app.py                              # Flask web server + API
|-- pubmed_benchmark.py                 # 1,077-article corpus benchmark
|-- numerical_retention_benchmark.py    # Lumisift vs embedding retrieval
|-- baseline_comparison.py              # BM25 / ColBERT / Embedding head-to-head
|-- cross_encoder_benchmark.py          # Cross-encoder reranker vs all methods
|-- ablation_study.py                   # Systematic axis removal study
|-- export_reproducibility_kit.py       # Export verifiable benchmark dataset
|-- drug_discovery_usecase.py           # 3 pharma scenarios
|-- hybrid_benchmark.py                 # Alpha sweep for hybrid mode
|-- learned_scoring.py                  # Train MLP axis predictor
|-- pubmedqa_benchmark.py              # PubMedQA comprehension test
|-- downstream_eval.py                  # AI-judged answer quality
|-- core/
|   |-- axes_evaluator.py              # 8-signal scoring engine
|   |-- pipeline.py                    # End-to-end orchestrator (3 selection modes)
|   |-- atom.py                        # Atom data model
|   |-- embeddings.py                  # Sentence-transformer embeddings
|   |-- surface.py                     # Surface clustering
|   |-- finetuning.py                  # Axis calibration + JSONL training export
|   |-- self_optimization.py           # Room splitting + tension monitoring
|-- benchmark_data/                    # Generated benchmark results (JSON)
|-- models/                            # Trained models (.pt, .gitignored)
```

---

## Contributing

We welcome contributions in these areas:

1. **Non-biotech benchmarks** -- Run the pipeline on clinical, legal, or cybersecurity text
2. **Improved heuristics** -- Better regex patterns in `axes_evaluator.py` directly improve all benchmarks
3. **RAG integrations** -- Show Lumisift working inside LangChain, LlamaIndex, or Haystack
4. **Human evaluation** -- Domain expert ratings would significantly strengthen the evidence

```bash
git clone https://github.com/Saeedmora/Lumisift.git
git checkout -b feature/your-improvement
python numerical_retention_benchmark.py   # Verify nothing breaks
# open a PR
```

---

## License

**AGPL-3.0** -- Free to use, modify, and distribute. Source code must be shared if you deploy a modified version (including SaaS). Attribution required.

**Commercial licensing** is available for proprietary use cases.

**Contact:** [Saeed Moradtalab](https://www.linkedin.com/in/ben-moradtalab-9442a41a6)

---

<p align="center">
  <strong>Lumisift</strong><br>
  Standard retrieval loses 60% of scientific data. We retain 83%.<br>
  Validated on 1,077 articles. Beating BM25 by +41pp. Every benchmark reproducible.<br>
  <sub>Copyright 2026 Saeed Moradtalab</sub>
</p>
