<p align="center">
  <strong>Lumisift</strong><br>
  <em>Multi-Axis Semantic Intelligence for Scientific Text</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0"></a>
  <a href="#benchmark"><img src="https://img.shields.io/badge/benchmark-95_PubMed_articles-orange?style=flat-square" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-not_required-brightgreen?style=flat-square" alt="No GPU"></a>
</p>

<p align="center">
  <strong>The first step toward the Autonomous Agentic Scientist.</strong><br>
  <sub>Created by <a href="https://www.linkedin.com/in/ben-moradtalab-9442a41a6">Saeed Moradtalab</a></sub>
</p>

---

## What is Lumisift?

**Lumisift is a pre-analysis layer that evaluates scientific text across 8 semantic dimensions -- before your AI ever sees it.**

Standard RAG retrieves text by similarity.
Lumisift scores every passage on **Relevance, Risk, Trust, Causality, Temporality, Visibility, Ontology, and Specificity** -- then selects only what matters.

```
Raw Text --> Embedding --> 8-Axis Evaluation --> Intelligent Selection --> LLM
                               |
                     Not "what looks similar"
                     but "what IS important"
```

> *"This is the first step toward the **Autonomous Agentic Scientist** -- an AI that reads,
> evaluates, prioritizes, and reasons about scientific literature on its own."*
>
> -- Saeed Moradtalab

---

## Why Lumisift?

| Problem | Without Lumisift | With Lumisift |
|---------|----------------|--------------|
| Context selection | Similarity only (1 dimension) | **8 semantic axes** |
| Quantitative data | Often discarded | **Specificity boost** keeps numbers |
| Text fidelity | Lossy (summarization) | **100% lossless** -- original text preserved |
| Cost | Full context = full price | **52% fewer tokens** = 52% cheaper |
| Privacy | Cloud APIs required | **100% local** -- no data leaves your machine |
| Customization | None | **Self-calibrating** -- learns from your corrections |

---

## Verified Results

Benchmarked on **95 peer-reviewed PubMed protein engineering articles**. All numbers reproducible.

| Metric | Value |
|--------|-------|
| Context reduction | **51.8%** avg (up to 82.7%) |
| Downstream quality | **60%** of articles retain perfect AI answer quality |
| Accuracy (AI judge) | **3.9 / 5.0** |
| Completeness | **3.9 / 5.0** (up from 2.8 after specificity boost) |
| Efficiency gain | **+63.2%** quality per token vs full text |
| Processing speed | **4.9 articles/sec** on CPU |
| Ontology coverage | **100%** (0% unknown) |
| Training data | **352 labeled samples** auto-generated |

<details>
<summary><strong>How we measured (click to expand)</strong></summary>

1. Generated scientific questions from full abstracts (Gemini 3 Flash Preview)
2. Answered each question with full text and Lumisift-selected text
3. Blind grading by AI judge on Accuracy, Completeness, Relevance, Conciseness (1-5)
4. **Reproducible:** `python downstream_eval.py`

</details>

---

## The 8 Semantic Axes

| Axis | What it measures | Range | Example use |
|------|-----------------|-------|-------------|
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

Chunks with mutation rates, IC50 values, fold-changes, and p-values get up to **1.8x priority** -- they won't be discarded.

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
|  Raw Text --> Embedding --> 8-Axis Evaluation --> Atom         |
|               (MiniLM)     (Heuristic / TinyLlama / NF4)      |
|                                                                |
|  Atoms --> Surface Clustering --> Room Assignment              |
|            (similarity-based)     (self-optimizing)             |
|                                                                |
|  Selection: Multi-axis scoring with specificity boost          |
|             Top-k atoms -> Optimized context for LLM           |
|                                                                |
|  Calibration: User feedback -> Axis weight adjustment          |
|               JSONL export for LoRA / QLoRA training           |
+---------------------------------------------------------------+
```

### Three Evaluator Modes

| Mode | Backend | Speed | Quality | Requires |
|------|---------|-------|---------|----------|
| **Heuristic** | Keyword + regex | ~0ms/chunk | Good | Nothing |
| **GGUF Q4** | TinyLlama 1.1B (4-bit) | ~200ms/chunk | Better | `models/*.gguf` |
| **NF4** | HuggingFace + bitsandbytes | ~150ms/chunk | Best | CUDA GPU |

Auto-selects the best available mode. Falls back gracefully.

---

## Who Is This For?

| User | Use Case |
|------|----------|
| **Researchers** | Triage 200+ papers by relevance, trust, and causality -- not just keyword similarity |
| **Biotech / Pharma** | Protect quantitative data (IC50, fold-changes, mutation rates) during context compression |
| **AI Engineers** | Generate structured training data (8-axis labels) automatically from any text corpus |
| **Security Teams** | Triage threat advisories by risk x trust -- separate zero-days from routine patches |
| **Legal / Compliance** | Identify causal obligations in regulatory documents |

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
├── app.py                     # Flask web server + API
├── pubmed_benchmark.py        # PubMed corpus benchmark
├── downstream_eval.py         # AI quality evaluation
├── core/
│   ├── axes_evaluator.py      # 8-axis evaluation (heuristic / LLM / NF4)
│   ├── nf4_loader.py          # NF4 quantization + embedding compression
│   ├── pipeline.py            # End-to-end orchestrator
│   ├── atom.py                # Atom data model
│   ├── surface.py             # Surface clustering
│   ├── finetuning.py          # Axis calibration + training export
│   ├── embeddings.py          # Sentence-transformer embeddings
│   └── self_optimization.py   # Room splitting + tension monitoring
├── static/
│   └── index.html             # Web UI
├── benchmark_data/            # Generated results
└── models/                    # Local models (optional)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run the benchmark: `python pubmed_benchmark.py`
4. Submit a pull request

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
  <strong>Lumisift</strong> -- Not compression. Selection.<br>
  <sub>Copyright 2026 Saeed Moradtalab</sub>
</p>
