<p align="center">
  <strong>ðŸ”¬ AxiSift</strong><br>
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

## What is AxiSift?

**AxiSift is a pre-analysis layer that evaluates scientific text across 8 semantic dimensions â€” before your AI ever sees it.**

Standard RAG retrieves text by similarity.  
AxiSift scores every passage on **Relevance, Risk, Trust, Causality, Temporality, Visibility, Ontology, and Specificity** â€” then selects only what matters.

```
Raw Text â†’ Embedding â†’ 8-Axis Evaluation â†’ Intelligent Selection â†’ LLM
                              â†‘
                    Not "what looks similar"
                    but "what IS important"
```

> _"This is the first step toward the **Autonomous Agentic Scientist** â€” an AI that reads,
> evaluates, prioritizes, and reasons about scientific literature on its own."_
>
> â€” Saeed Moradtalab

---

## Why AxiSift?

| Problem | Without AxiSift | With AxiSift |
|---------|----------------|--------------|
| Context selection | Similarity only (1 dimension) | **8 semantic axes** |
| Quantitative data | Often discarded | **Specificity boost** keeps numbers |
| Text fidelity | Lossy (summarization) | **100% lossless** â€” original text preserved |
| Cost | Full context = full price | **52% fewer tokens** = 52% cheaper |
| Privacy | Cloud APIs required | **100% local** â€” no data leaves your machine |
| Customization | None | **Self-calibrating** â€” learns from your corrections |

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
2. Answered each question with full text and AxiSift-selected text
3. Blind grading by AI judge on Accuracy, Completeness, Relevance, Conciseness (1â€“5)
4. **Reproducible:** `python downstream_eval.py`

</details>

---

## The 8 Semantic Axes

| Axis | What it measures | Range | Example use |
|------|-----------------|-------|-------------|
| **Temporal** | Currency of information | âˆ’1 â†’ +1 | Filter outdated findings |
| **Relevance** | Strategic importance | 0 â†’ 1 | Prioritize key results |
| **Risk** | Uncertainty level | âˆ’1 â†’ +1 | Flag preliminary data |
| **Ontology** | Domain category | 0 â†’ 1 | Classify by topic |
| **Causality** | Cause vs. effect | âˆ’1 â†’ +1 | Map causal chains |
| **Visibility** | Internal vs. public | 0 â†’ 1 | Control data exposure |
| **Trust** | Source reliability | 0 â†’ 1 | Weight verified vs. unverified |
| **Specificity** | Quantitative data density | 0 â†’ 1 | Protect numbers, rates, measurements |

### The Selection Formula

```
score = relevance Ã— (1 + |risk|) Ã— (0.5 + trust Ã— 0.5) Ã— temporal_boost Ã— specificity_boost
```

Chunks with mutation rates, IC50 values, fold-changes, and p-values get up to **1.8Ã— priority** â€” they won't be discarded.

---

## Quick Start

```bash
# Clone
git clone https://github.com/your-username/AxiSift.git
cd AxiSift

# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e .

# Run
python app.py
# â†’ http://localhost:5000
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
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      AxiSift PIPELINE                         â”‚
â”‚                                                               â”‚
â”‚  Raw Text â”€â”€â†’ Embedding â”€â”€â†’ 8-Axis Evaluation â”€â”€â†’ Atom       â”‚
â”‚               (MiniLM)      (Heuristic / TinyLlama / NF4)    â”‚
â”‚                                                               â”‚
â”‚  Atoms â”€â”€â†’ Surface Clustering â”€â”€â†’ Room Assignment             â”‚
â”‚            (similarity-based)     (self-optimizing)            â”‚
â”‚                                                               â”‚
â”‚  Selection: Multi-axis scoring with specificity boost          â”‚
â”‚             Top-k atoms â†’ Optimized context for LLM           â”‚
â”‚                                                               â”‚
â”‚  Calibration: User feedback â†’ Axis weight adjustment           â”‚
â”‚               JSONL export for LoRA / QLoRA training           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
| **Researchers** | Triage 200+ papers by relevance, trust, and causality â€” not just keyword similarity |
| **Biotech / Pharma** | Protect quantitative data (IC50, fold-changes, mutation rates) during context compression |
| **AI Engineers** | Generate structured training data (8-axis labels) automatically from any text corpus |
| **Security Teams** | Triage threat advisories by risk Ã— trust â€” separate zero-days from routine patches |
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
AxiSift/
â”œâ”€â”€ app.py                     # Flask web server + API
â”œâ”€â”€ pubmed_benchmark.py        # PubMed corpus benchmark
â”œâ”€â”€ downstream_eval.py         # AI quality evaluation
â”œâ”€â”€ core/
â”‚   â”œâ”€â”€ axes_evaluator.py      # 8-axis evaluation (heuristic / LLM / NF4)
â”‚   â”œâ”€â”€ nf4_loader.py          # NF4 quantization + embedding compression
â”‚   â”œâ”€â”€ pipeline.py            # End-to-end orchestrator
â”‚   â”œâ”€â”€ atom.py                # Atom data model
â”‚   â”œâ”€â”€ surface.py             # Surface clustering
â”‚   â”œâ”€â”€ finetuning.py          # Axis calibration + training export
â”‚   â”œâ”€â”€ embeddings.py          # Sentence-transformer embeddings
â”‚   â””â”€â”€ self_optimization.py   # Room splitting + tension monitoring
â”œâ”€â”€ static/
â”‚   â””â”€â”€ index.html             # Web UI
â”œâ”€â”€ benchmark_data/            # Generated results
â””â”€â”€ models/                    # Local models (optional)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run the benchmark: `python pubmed_benchmark.py`
4. Submit a pull request

---

## License

**AGPL-3.0** â€” see [LICENSE](LICENSE) for the full text.

This project is open source under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

- âœ… Free to use, modify, and distribute â€” keep the same license
- âœ… Source code must be shared if you deploy a modified version (even as SaaS)
- âœ… Attribution required

### Commercial Licensing

Want to use AxiSift in a **proprietary product** without AGPL obligations?  
A commercial license is available.

ðŸ“§ **Saeed Moradtalab**  
ðŸ”— [LinkedIn](https://www.linkedin.com/in/ben-moradtalab-9442a41a6)

---

<p align="center">
  <strong>AxiSift</strong> â€” Not compression. Selection.<br>
  <sub>Â© 2026 Saeed Moradtalab</sub>
</p>

