<p align="center">
  <strong>Lumisift</strong><br>
  <em>Multi-Axis Context Selection for Scientific AI Pipelines</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0"></a>
  <a href="#benchmark"><img src="https://img.shields.io/badge/benchmark-95_PubMed_articles-orange?style=flat-square" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-not_required-brightgreen?style=flat-square" alt="No GPU"></a>
</p>

<p align="center">
  A multi-signal pre-filter for LLM context windows.<br>
  <sub>Created by <a href="https://www.linkedin.com/in/ben-moradtalab-9442a41a6">Saeed Moradtalab</a></sub>
</p>

---

## The Problem

Standard RAG retrieves text by **vector similarity** -- but similarity is not importance.

A sentence describing experimental setup may be *highly similar* to your query but contain **zero useful data**. The sentence with the actual mutation rate (7.2 x 10^-5) may *not look similar at all* -- it's just a number in a results paragraph.

Lumisift addresses this by scoring text across **8 independent semantic signals** before selecting what to send to the LLM.

```
Raw Text --> Embedding --> 8-Signal Evaluation --> Intelligent Selection --> LLM
                               |
                     Not "what looks similar"
                     but "what IS important"
```

> **What this is:** A smarter context selection layer that sits between your data and your LLM.
>
> **What this is not:** A complete autonomous research system. It's one component -- the filtering layer -- that makes downstream LLM interactions more efficient.

---

## What Lumisift Actually Does

| Problem | Standard RAG | With Lumisift |
|---------|-------------|--------------|
| Context selection | Similarity only (1 dimension) | **8 semantic signals** |
| Quantitative data | Often discarded by similarity search | **Specificity boost** preserves numbers |
| Text fidelity | Some systems summarize (lossy) | **100% lossless** -- original text preserved |
| Token cost | Full context = full price | **~52% fewer tokens** sent to LLM |
| Privacy | Often requires cloud APIs | **Runs 100% locally** -- no data leaves your machine |

### What We Claim (and What We Don't)

**We claim:**
- Multi-axis scoring catches signals that pure similarity misses
- Token reduction is measurable and reproducible
- The system works locally without GPU

**We do NOT claim:**
- That this replaces proper retrieval systems (it complements them)
- That 95 articles proves generalization across all domains
- That heuristic scoring is production-grade for all use cases (see [Limitations](#limitations))

---

## Benchmark Results

Benchmarked on **95 peer-reviewed PubMed protein engineering articles**. This is early validation, not exhaustive proof.

| Metric | Value | Note |
|--------|-------|------|
| Context reduction | **51.8%** avg | Best: 82.7%, Worst: 0% (single-atom articles) |
| Downstream quality | **60%** retain perfect QA quality | Measured via Gemini 3 Flash as judge |
| Accuracy score | **3.9 / 5.0** | AI judge, not human evaluation |
| Completeness | **3.9 / 5.0** | Up from 2.8 after specificity boost |
| Efficiency gain | **+63.2%** quality per token | Compared to sending full text |
| Processing speed | **4.9 articles/sec** | CPU only, no GPU |
| Ontology coverage | **100%** | 0% unknown after keyword expansion |

<details>
<summary><strong>How we measured (click to expand)</strong></summary>

1. Generated scientific questions from full abstracts (Gemini 3 Flash Preview)
2. Answered each question with full text and Lumisift-selected text
3. Blind grading by AI judge on Accuracy, Completeness, Relevance, Conciseness (1-5)
4. **Reproducible:** `python downstream_eval.py`

**Known limitations of this benchmark:**
- 95 articles is a small corpus -- larger validation needed
- AI judge introduces subjectivity -- human evaluation would be stronger
- Only tested on protein engineering -- cross-domain validation is pending
- "Perfect quality" means 20/20 on the composite score, which is a high but specific bar

</details>

### Head-to-Head: Lumisift vs Embedding Retrieval

First direct comparison. Same articles, same 50% selection ratio, different method.

| Method | Numerical Facts Retained | Rate |
|--------|------------------------|------|
| Embedding Similarity (standard RAG) | 28 / 77 | **36.4%** |
| Lumisift (8-axis + specificity) | 68 / 77 | **88.3%** |
| **Delta** | **+40 facts** | **+51.9 pp** |

**Standard embedding retrieval loses 63.6% of quantitative data. Lumisift loses 11.7%.**

| Fact Type | Embedding | Lumisift |
|-----------|-----------|----------|
| Fold changes (e.g. "1000-fold") | 22.2% | **88.9%** |
| Precise decimals | 22.2% | **100%** |
| Concentrations (e.g. "50 mM") | 28.6% | **100%** |
| Percentages | 60.0% | **80.0%** |

Per-article: Lumisift wins 61%, Embedding wins 4%, Ties 36%.

**Reproducible:** `python numerical_retention_benchmark.py`

### Drug Discovery Use Case

Three real-world pharma scenarios (EGFR inhibitor, lipase evolution, mRNA LNP optimization):

| Scenario | Embedding | Lumisift |
|----------|-----------|----------|
| EGFR inhibitor (IC50, TGI, selectivity) | 44% | **67%** |
| Lipase evolution (kcat/Km, E-value, ee%) | 0% | **86%** |
| mRNA LNP optimization (fold-change, ED50, PDI) | 0% | **100%** |
| **Average** | **15%** | **84%** |

**In drug discovery, embedding retrieval retains only 15% of critical data. Lumisift retains 84%.**

**Reproducible:** `python drug_discovery_usecase.py`

### PubMedQA-Style Benchmark (Honest Limitation)

Yes/no/maybe scientific questions (15 articles):

| Method | Accuracy |
|--------|----------|
| Full Context (100% tokens) | **93.3%** |
| Embedding Similarity (50%) | **93.3%** |
| Lumisift (50%) | **46.7%** |

Lumisift's specificity boost prioritizes quantitative chunks over explanatory context.
For factual yes/no questions, explanatory text matters more than numbers.

**Takeaway:** Lumisift excels at preserving quantitative data but should be combined with
similarity retrieval for comprehension tasks. It's a complement, not a replacement.

**Reproducible:** `python pubmedqa_benchmark.py`

---

## The 8 Semantic Signals

| Signal | What it measures | Range | Example use |
|--------|-----------------|-------|-------------|
| **Temporal** | Currency of information | -1 to +1 | Filter outdated findings |
| **Relevance** | Strategic importance | 0 to 1 | Prioritize key results |
| **Risk** | Uncertainty level | -1 to +1 | Flag preliminary data |
| **Ontology** | Domain category | 0 to 1 | Classify by topic |
| **Causality** | Cause vs. effect | -1 to +1 | Map causal chains |
| **Visibility** | Internal vs. public | 0 to 1 | Control data exposure |
| **Trust** | Source reliability | 0 to 1 | Weight verified vs. unverified |
| **Specificity** | Quantitative data density | 0 to 1 | Protect numbers, rates, measurements |

### The Selection Formula

```
score = relevance * (1 + |risk|) * (0.5 + trust * 0.5) * temporal_boost * specificity_boost
```

Chunks with mutation rates, IC50 values, fold-changes, and p-values get up to **1.8x priority**.

---

## Limitations

Being honest about what needs work:

| Limitation | Impact | Plan |
|-----------|--------|------|
| **Heuristic scoring** | Trust and causality are hard to infer via regex/keywords. Edge cases will break. | Replace with learned models as training data grows |
| **Small benchmark** | 95 articles in one domain. Not enough to prove generalization. | Expand to clinical, legal, and security corpora |
| **AI judge** | Gemini-as-evaluator introduces subjectivity and noise. | Add human evaluation protocol |
| **No baseline comparison** | Not benchmarked against BM25, ColBERT, or other retrieval systems. | Build comparison benchmark |
| **Domain transfer** | Keyword lexicon is tuned for biotech. Other domains will need calibration. | User feedback loop enables domain adaptation |

**In short:** This is a promising engineering tool with early validation, not a scientifically proven system. We're building in the open because we think the direction is right.

---

## Quick Start

```bash
# Clone
git clone https://github.com/Saeedmora/Lumisift.git
cd Lumisift

# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e .

# Run
python app.py
# Open http://localhost:5000
```

### Python API

```python
from core.pipeline import LogicalRoomsPipeline

pipe = LogicalRoomsPipeline(verbose=True)

# Process a single text
atom = pipe.process(
    "Directed evolution achieves 1000-fold improvement in catalytic activity.",
    domain="biotech"
)

print(f"Relevance:   {atom.axes['relevance']:.2f}")
print(f"Specificity: {atom.axes['specificity']:.2f}")
print(f"Trust:       {atom.axes['trust']:.2f}")
```

### Batch Processing & Selection

```python
papers = [
    "CRISPR-Cas9 enables precise genome editing in mammalian cells.",
    "Machine learning predicts protein stability with 85% accuracy.",
    "Novel lipid nanoparticles improve mRNA delivery efficiency by 300%.",
]

atoms = pipe.process_batch(papers, domain="biotech")

# Intelligent selection: keep only the most important passages
result = pipe.select_context(atoms, top_k=2)
print(f"Reduction: {result.compression_ratio:.1%}")
```

### Run the Benchmark

```bash
python pubmed_benchmark.py        # 95-article corpus benchmark
python downstream_eval.py         # AI quality evaluation (requires GEMINI_API_KEY)
```

---

## Architecture

```
+---------------------------------------------------------------+
|                      LUMISIFT PIPELINE                         |
|                                                                |
|  Raw Text --> Embedding --> 8-Signal Scoring --> Atom          |
|               (MiniLM)     (Heuristic / TinyLlama / NF4)      |
|                                                                |
|  Atoms --> Surface Clustering --> Room Assignment              |
|            (similarity-based)     (self-optimizing)             |
|                                                                |
|  Selection: Multi-signal scoring with specificity boost        |
|             Top-k atoms -> Optimized context for LLM           |
|                                                                |
|  Calibration: User feedback -> Axis weight adjustment          |
|               JSONL export for LoRA / QLoRA training           |
+---------------------------------------------------------------+
```

### Three Evaluator Modes

| Mode | Backend | Speed | Quality | Requires |
|------|---------|-------|---------|----------|
| **Heuristic** | Keyword + regex | ~0ms/chunk | Baseline | Nothing |
| **GGUF Q4** | TinyLlama 1.1B (4-bit) | ~200ms/chunk | Better | `models/*.gguf` |
| **NF4** | HuggingFace + bitsandbytes | ~150ms/chunk | Best | CUDA GPU |

Auto-selects the best available mode. Falls back gracefully.

---

## Roadmap

What would make this project actually important:

- [ ] **Learned scoring models** -- replace heuristics with trained classifiers
- [ ] **Large-scale validation** -- 1000+ articles across multiple domains
- [ ] **Baseline comparison** -- head-to-head vs BM25, ColBERT, and standard RAG
- [ ] **Human evaluation** -- expert ratings alongside AI judge
- [ ] **RAG integration** -- plug into LangChain/LlamaIndex as a re-ranker
- [ ] **Cross-domain testing** -- clinical, legal, cybersecurity corpora

Contributions welcome. See below.

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | Cloud evaluation via Google Gemini |
| `MODEL_PATH` | Optional | Path to local GGUF model |

Without API keys, everything runs **100% locally** with the heuristic evaluator.

---

## Project Structure

```
Lumisift/
|-- app.py                     # Flask web server + API
|-- pubmed_benchmark.py        # PubMed corpus benchmark
|-- downstream_eval.py         # AI quality evaluation
|-- core/
|   |-- axes_evaluator.py      # 8-signal scoring (heuristic / LLM / NF4)
|   |-- nf4_loader.py          # NF4 quantization + embedding compression
|   |-- pipeline.py            # End-to-end orchestrator
|   |-- atom.py                # Atom data model
|   |-- surface.py             # Surface clustering
|   |-- finetuning.py          # Axis calibration + training export
|   |-- embeddings.py          # Sentence-transformer embeddings
|   |-- self_optimization.py   # Room splitting + tension monitoring
|-- static/
|   |-- index.html             # Web UI
|-- benchmark_data/            # Generated results
|-- models/                    # Local models (optional)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run the benchmark: `python pubmed_benchmark.py`
4. Submit a pull request

We especially welcome:
- Benchmark contributions on non-biotech domains
- Improvements to the heuristic scoring rules
- Integration examples with existing RAG frameworks

---

## License

**AGPL-3.0** -- see [LICENSE](LICENSE) for the full text.

This project is open source under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

- Free to use, modify, and distribute -- keep the same license
- Source code must be shared if you deploy a modified version (even as SaaS)
- Attribution required

### Commercial Licensing

Want to use Lumisift in a **proprietary product** without AGPL obligations?
A commercial license is available.

**Saeed Moradtalab**
[LinkedIn](https://www.linkedin.com/in/ben-moradtalab-9442a41a6)

---

<p align="center">
  <strong>Lumisift</strong> -- Multi-signal context selection for scientific AI.<br>
  <sub>Copyright 2026 Saeed Moradtalab</sub>
</p>
