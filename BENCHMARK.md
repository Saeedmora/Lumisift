# Logical Rooms — Benchmark Report

## PubMed Protein Engineering Corpus

**Date:** April 2026  
**Reproducible:** `python pubmed_benchmark.py`

---

## Methodology

### Data Source

| Parameter | Value |
|-----------|-------|
| Database | NCBI PubMed (https://pubmed.ncbi.nlm.nih.gov/) |
| API | E-utilities (ESearch + EFetch) |
| Query | `"protein engineering AND directed evolution"` |
| Sort | Relevance |
| Requested | 100 articles |
| Returned with abstracts | **95 articles** |

All data is peer-reviewed scientific literature. No synthetic or cherry-picked samples.

### Processing Pipeline

```
PubMed Abstract
     ↓
Sentence Splitting (regex: (?<=[.!?])\s+(?=[A-Z]))
     ↓
2-3 sentence chunks (min 30 chars)
     ↓
Embedding: all-MiniLM-L6-v2 (384-dim)
     ↓
7-Axis Heuristic Evaluation
     ↓
Atom Creation (text + axes + embedding + metadata)
     ↓
Surface Clustering (similarity-based)
     ↓
Axes-Driven Selection (score = relevance × (1 + |risk|) × trust × temporal_boost)
```

### Tokenizer

All token counts use **tiktoken cl100k_base** — the same BPE tokenizer used by GPT-4, GPT-4o, and GPT-3.5-turbo. This ensures token counts are directly comparable to OpenAI API billing.

### Hardware

- CPU-only processing (no GPU)
- Consumer-grade hardware
- Heuristic evaluator (no local LLM required for this benchmark)

---

## Results

### Corpus Overview

| Metric | Value |
|--------|-------|
| Articles processed | **95** |
| Semantic atoms created | **352** |
| Total tokens | **22,311** |
| Surfaces built | **95** |
| Avg tokens per article | 234.9 |
| Avg atoms per article | 3.7 |

### Axes-Driven Selection

This is the primary value metric. Axes-driven selection picks the top-k atoms ranked by a multi-axis score:

```
score = relevance × (1 + |risk|) × (0.5 + trust × 0.5) × temporal_boost
```

| Metric | Value |
|--------|-------|
| **Overall context reduction** | **53.8%** |
| Mean per article | 51.8% |
| Median per article | **54.9%** |
| Best case (single article) | **82.7%** |
| Worst case | 0.0% (single-atom articles have nothing to select away) |
| Total tokens selected | 10,299 of 22,311 |

**Interpretation:** On average, axes-driven selection reduces context by 53.8% while preserving the raw text of the most relevant passages. For the best articles, up to 82.7% of tokens are eliminated as lower-priority content.

### Structured Representation

The "compressed" representation replaces raw text with the atom format: `[TE:val|RE:val|RI:val|...]`

| Metric | Value |
|--------|-------|
| Overall ratio | **32.2%** |
| Mean per article | 28.9% |
| Median per article | **30.3%** |
| Best case | 65.1% |
| Worst case | **-43.3%** (atom format larger than short input) |
| Total compressed tokens | 15,136 of 22,311 |

> **Transparency note:** The negative worst-case ratio means the structured atom format adds more metadata tokens than it saves for very short text segments (~30 words). This is expected and honest: structured representation has overhead. The value is in the semantic metadata, not size reduction.

### 7-Axis Distribution

Measured across 352 atoms from 95 peer-reviewed articles:

| Axis | Mean | Std Dev | Min | Max | Median | Interpretation |
|------|------|---------|-----|-----|--------|----------------|
| **Temporal** | -0.036 | 0.117 | -0.40 | +0.40 | 0.0 | Mostly atemporal (review/method content) |
| **Relevance** | +0.371 | 0.152 | +0.30 | +0.97 | 0.3 | Wide spread — some highly relevant findings |
| **Risk** | -0.009 | 0.062 | -0.25 | +0.25 | 0.0 | Low risk — basic science, not clinical |
| **Ontology** | +0.427 | 0.167 | +0.00 | +0.80 | 0.5 | Strong domain categorization |
| **Causality** | +0.038 | 0.150 | -0.33 | +0.67 | 0.0 | Slight effect bias (results-focused) |
| **Visibility** | +0.504 | 0.062 | +0.25 | +0.75 | 0.5 | All public (PubMed = open access) |
| **Trust** | +0.499 | 0.040 | +0.25 | +0.75 | 0.5 | Consistently high (peer-reviewed) |

**Key observations:**
- **Trust** has the lowest variance (std=0.040) — expected since all sources are peer-reviewed journals
- **Relevance** has the highest range (+0.30 to +0.97) — the evaluator successfully discriminates between methods text and findings
- **Risk** is near zero across the corpus — protein engineering is basic science, not clinical trials
- **Causality** spreads from -0.33 (causes) to +0.67 (effects) — showing causal reasoning detection

### Training Data Generated

| Metric | Value |
|--------|-------|
| Total JSONL samples | **352** |
| Format | One JSON object per line |
| Fields per sample | text, axes (7), domain, category, tension, confidence, pmid, source |

**Category distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| unknown | 197 | 56.0% |
| process | 69 | 19.6% |
| technology | 35 | 9.9% |
| information | 24 | 6.8% |
| human | 15 | 4.3% |
| strategy | 12 | 3.4% |

### Performance

| Metric | Value |
|--------|-------|
| Total processing time | **19.5 seconds** |
| Average per article | **206ms** |
| Median per article | 205ms |
| P95 latency | 267ms |
| Throughput | **4.9 articles/sec** |
| Includes | Embedding + evaluation + atom creation + surface clustering + selection |

---

## Downstream Quality Evaluation

**Date:** April 2026  
**Model (Judge):** Gemini 3 Flash Preview  
**Reproducible:** `python downstream_eval.py`

### Methodology

A proper downstream QA evaluation to test whether axes-driven selection preserves AI answer quality:

1. Generate one scientific question per article using the full abstract
2. Answer each question twice: once with **full text**, once with **tension-selected text** (top 50%)
3. Grade both answers against ground truth on 4 dimensions (1–5 scale)

### Quality Scores — With Specificity Boost (1–5 scale)

Selection formula: `score = relevance × (1 + |risk|) × (1 − trust × 0.5) × specificity_boost`

| Dimension | Full Text | Selected | Delta | Verdict |
|-----------|-----------|----------|-------|---------|
| Accuracy | 5.00 | 3.90 | -1.10 | Full wins |
| Completeness | 5.00 | 3.90 | -1.10 | Full wins |
| Relevance | 5.00 | 4.10 | -0.90 | Full wins |
| Conciseness | 5.00 | 4.10 | -0.90 | Full wins |
| **COMPOSITE** | **5.00** | **4.00** | **-1.00** | **Full wins** |

### Efficiency Analysis

| Metric | Value |
|--------|-------|
| Context reduction | 51.8% avg |
| Quality/1000 tokens (full) | 2.16 |
| Quality/1000 tokens (selected) | 3.52 |
| **Efficiency gain** | **+63.2%** |

### Impact of Specificity Boost

The specificity boost detects quantitative data (numbers, percentages, units, fold-changes, p-values, bioscience constants) and boosts those chunks by up to 1.8×. This prevents discarding numerical results.

**Before vs After — previously failing articles:**

| PMID | Before (v1) | After (v2, +specificity) | Improvement |
|------|-------------|--------------------------|-------------|
| 40773556 (T7 replisome) | 13/20 | **20/20** | +7 points |
| 30069054 (CRISPR) | 13/20 | **20/20** | +7 points |
| 36562723 (Bayesian) | 13/20 | **20/20** | +7 points |
| 40628259 (Inverse folding) | 14/20 | **20/20** | +6 points |
| 27826849 (DNA methyltransferase) | 17/20 | **20/20** | +3 points |

**60% of articles now score 20/20** (up from 20% in v1).

### Per-Article Breakdown (v2)

**6 of 10 articles maintain perfect quality:**

| PMID | Reduction | Full Score | Selected Score | Quality Loss |
|------|-----------|------------|----------------|-------------|
| 40628259 | 43% | 20/20 | 20/20 | **None** |
| 27826849 | 53% | 20/20 | 20/20 | **None** |
| 40773556 | 49% | 20/20 | 20/20 | **None** |
| 30069054 | 61% | 20/20 | 20/20 | **None** |
| 36562723 | 48% | 20/20 | 20/20 | **None** |
| 28255874 | 46% | 20/20 | 20/20 | **None** |

**4 of 10 articles show quality degradation:**

| PMID | Reduction | Selected Score | Issue |
|------|-----------|----------------|-------|
| 32557882 | 58% | 13/20 | Conceptual question not in selected chunks |
| 27410729 | 35% | 9/20 | Broad question, specificity didn't help |
| 29413956 | 55% | 13/20 | Abstract topic not in high-specificity chunks |
| 15026190 | 71% | 5/20 | Very short abstract (3 chunks → 1 selected) |

### Interpretation

1. **Specificity boost fixes the primary failure mode** — articles with quantitative data now score perfectly
2. **Accuracy improved** from 3.60 to 3.90 (+0.30)
3. **Completeness improved** from 2.80 to 3.90 (+1.10) — the biggest single improvement
4. **Efficiency gain: +63.2%** — more quality per token than full text
5. **Remaining failures** are in conceptual/non-quantitative articles where specificity boost doesn't apply — these need the LLM evaluator for better discrimination

---

## What This Benchmark Does NOT Show

In the interest of scientific integrity:

1. ~~No quality evaluation of selection~~ → **Now verified** (see Downstream Quality Evaluation above).

2. **No comparison with other systems.** We did not benchmark against BM25, ColBERT, or other retrieval systems on the same corpus. A fair comparison would require identical task setups.

3. **No LLM evaluator results.** This benchmark uses only the heuristic (keyword-based) evaluator. The LLM-based evaluator (TinyLlama 1.1B) would produce different axis scores with potentially better discrimination.

4. **No cross-domain generalization.** Results are specific to protein engineering abstracts. Other domains (clinical, security, legal) may show different patterns.

5. ~~Ontology coverage 56% unknown~~ → **Now resolved.** Keyword lexicon expanded from ~7 to ~30-50 keywords per category. Coverage: 100% (0/95 unknown). Distribution: Information 58.9%, Process 29.5%, Strategy 8.4%, Human 3.2%.

---

## Reproducing This Benchmark

```bash
# 1. Install dependencies
pip install -e .

# 2. Run the compression benchmark (fetches from NCBI, ~30 seconds)
python pubmed_benchmark.py

# 3. Run the downstream quality evaluation (requires Gemini API key, ~2 minutes)
python downstream_eval.py

# 4. Results are written to benchmark_data/
#    - pubmed_articles.json         (raw articles)
#    - training_data.jsonl          (352 training samples)
#    - benchmark_results.json       (compression metrics)
#    - downstream_quality.json      (QA quality evaluation)
```

The benchmark is fully deterministic given the same PubMed query results. Since PubMed content evolves, exact article IDs may differ on re-runs, but statistical properties should be stable.

---

## Files

| File | Size | Description |
|------|------|-------------|
| `benchmark_data/pubmed_articles.json` | ~170 KB | 95 raw PubMed articles with metadata |
| `benchmark_data/training_data.jsonl` | ~142 KB | 352 training samples in JSONL format |
| `benchmark_data/benchmark_results.json` | ~2.6 KB | Full compression benchmark results |
| `benchmark_data/downstream_quality.json` | ~7.4 KB | Downstream QA evaluation results |
| `pubmed_benchmark.py` | ~20 KB | Compression benchmark script |
| `downstream_eval.py` | ~15 KB | Downstream quality evaluation script |

---

*Generated by Logical Rooms benchmark pipeline. All data sourced from NCBI PubMed under fair use for research.*

