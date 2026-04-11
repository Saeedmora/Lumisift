"""
BM25 & ColBERT Baseline Comparison
====================================
Head-to-head: BM25, ColBERT (late interaction), Embedding, Lumisift, Hybrid
on numerical retention across 1,077 PubMed articles.

Established baselines:
  - BM25: Classic keyword-based retrieval (Okapi BM25)
  - ColBERT-style: Late-interaction token-level matching via MiniLM
  - Embedding: Standard cosine similarity (MiniLM sentence embeddings)
  - Lumisift: 8-axis multi-signal scoring
  - Hybrid: alpha=0.3 blend of similarity + lumisift
"""

import os
import sys
import json
import re
import time
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService

# ─── Numerical Fact Extraction ─────────────────────────────────────────────

NUMERICAL_PATTERNS = [
    (r'\b\d+\.?\d*\s*%', "percentage"),
    (r'\b\d+\.?\d*[-\s]?fold\b', "fold_change"),
    (r'\b\d+\.?\d*[xX]\b', "fold_change"),
    (r'\b\d+\.?\d*\s*[xX]\s*10\^?[-+]?\d+', "scientific_notation"),
    (r'\b\d+\.?\d*[eE][-+]?\d+', "scientific_notation"),
    (r'\b(?:IC50|EC50|IC_50|EC_50)\s*[=:~]?\s*\d+\.?\d*\s*(?:nM|uM|mM|ng|mg|ug)', "ic50_ec50"),
    (r'\b(?:Kd|Km|kcat|Vmax)\s*[=:~]?\s*\d+\.?\d*', "kinetic_constant"),
    (r'\b[Pp]\s*[<>=]\s*0?\.\d+', "p_value"),
    (r'\b\d+\.?\d*\s*(?:mM|uM|nM|pM|mg/mL|ng/mL|ug/mL|g/L|mol/L|mg/kg)', "concentration"),
    (r'\b\d+\.?\d*\s*(?:hours?|hrs?|min(?:utes?)?|days?)\b', "duration"),
    (r'\b\d{3,}\b', "large_number"),
    (r'\b\d+\.\d{2,}\b', "precise_decimal"),
]


def extract_facts(text):
    facts = []
    seen = set()
    for pattern, ftype in NUMERICAL_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = m.group().strip()
            key = f"{ftype}:{val}"
            if key not in seen:
                seen.add(key)
                facts.append({"value": val, "type": ftype})
    return facts


def facts_retained(facts, text):
    return [f for f in facts if f["value"] in text]


def normalize_scores(scores):
    """Normalize array to 0-1 range."""
    scores = np.array(scores, dtype=float)
    smin, smax = scores.min(), scores.max()
    if smax - smin > 1e-8:
        return (scores - smin) / (smax - smin)
    return np.ones_like(scores) * 0.5


# ─── Selection Methods ────────────────────────────────────────────────────

def select_bm25(query, chunks, top_k):
    """BM25 keyword-based retrieval."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores


def select_colbert_style(query_emb_tokens, chunk_embs_tokens, top_k):
    """
    ColBERT-style late interaction: MaxSim between query tokens
    and document tokens. Uses word-level embeddings from MiniLM.
    
    For each query token, find its max similarity with any doc token.
    Document score = sum of MaxSim across all query tokens.
    """
    scores = []
    for chunk_embs in chunk_embs_tokens:
        if len(chunk_embs) == 0:
            scores.append(0.0)
            continue
        # MaxSim: for each query token, max similarity to any chunk token
        # query_emb_tokens: (q_len, dim), chunk_embs: (d_len, dim)
        sim_matrix = query_emb_tokens @ chunk_embs.T  # (q_len, d_len)
        max_sims = sim_matrix.max(axis=1)  # (q_len,)
        scores.append(float(max_sims.sum()))
    scores = np.array(scores)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores


def select_embedding(query_emb, chunk_embs, top_k):
    """Standard cosine similarity."""
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8
    c = chunk_embs / norms
    scores = c @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores


def select_lumisift(chunks, evaluator, top_k):
    """Lumisift 8-axis multi-signal scoring."""
    scores = []
    for chunk in chunks:
        axes, _ = evaluator.evaluate(chunk)
        rel = abs(axes.get("relevance", 0))
        risk = abs(axes.get("risk", 0))
        trust = axes.get("trust", 0.5)
        spec = axes.get("specificity", 0.0)
        s_boost = 1.0 + spec * 0.8
        score = rel * (1 + risk) * (0.5 + trust * 0.5) * s_boost
        scores.append(score)
    scores = np.array(scores)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores


def select_hybrid(sim_scores, lumi_scores, alpha, top_k):
    """Hybrid: alpha * similarity + (1-alpha) * lumisift."""
    sim_n = normalize_scores(sim_scores)
    lumi_n = normalize_scores(lumi_scores)
    hybrid = alpha * sim_n + (1 - alpha) * lumi_n
    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return top_idx, hybrid


# ─── Main Benchmark ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BM25 / ColBERT BASELINE COMPARISON")
    print("  Head-to-head: 5 methods on numerical retention")
    print("=" * 70)
    print()

    # Load articles
    articles_path = os.path.join("benchmark_data", "pubmed_articles.json")
    with open(articles_path, "r", encoding="utf-8") as f:
        all_articles = json.load(f)

    articles = [a for a in all_articles if len(a.get("abstract", "").split()) > 50]
    print(f"Loaded {len(articles)} articles\n")

    # Initialize
    print("Loading models...")
    embedder = EmbeddingService()
    evaluator = SevenAxesEvaluator(use_llm=False)
    print("Models ready.\n")

    # ─── Process all articles ──────────────────────────────────────────────

    methods = ["BM25", "ColBERT", "Embedding", "Lumisift", "Hybrid (a=0.3)"]
    method_stats = {m: {"retained": 0, "total": 0, "wins": 0, "articles": 0} for m in methods}
    type_stats = {}

    articles_processed = 0
    articles_with_facts = 0

    for i, article in enumerate(articles):
        abstract = article.get("abstract", "")
        title = article.get("title", "")[:60]

        # Split into chunks
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract)
        chunks = []
        current = ""
        for s in sentences:
            current += (" " if current else "") + s
            if len(current.split()) >= 20:
                chunks.append(current.strip())
                current = ""
        if current.strip():
            chunks.append(current.strip())

        if len(chunks) < 2:
            continue

        articles_processed += 1
        n_select = max(1, len(chunks) // 2)

        # Extract numerical facts
        all_facts = extract_facts(abstract)
        if not all_facts:
            continue

        articles_with_facts += 1

        # ── Compute all method scores ──────────────────────────────────

        # Query = article title (same for all methods)
        query = title

        # BM25
        bm25_idx, bm25_scores = select_bm25(query, chunks, n_select)

        # Embedding (sentence-level)
        query_emb = embedder.embed(query)
        chunk_embs = embedder.embed_many(chunks)
        emb_idx, emb_scores = select_embedding(query_emb, chunk_embs, n_select)

        # ColBERT-style (token-level late interaction)
        # Embed query words individually for token-level matching
        query_words = query.split()
        if len(query_words) < 2:
            query_words = query_words + ["research"]  # pad single-word queries
        query_word_embs = embedder.embed_many(query_words)
        if query_word_embs.ndim == 1:
            query_word_embs = query_word_embs.reshape(1, -1)
        query_word_embs = query_word_embs / (np.linalg.norm(query_word_embs, axis=1, keepdims=True) + 1e-8)

        chunk_word_embs = []
        for chunk in chunks:
            words = chunk.split()[:50]  # Limit for speed
            if words:
                w_embs = embedder.embed_many(words)
                w_embs = w_embs / (np.linalg.norm(w_embs, axis=1, keepdims=True) + 1e-8)
                chunk_word_embs.append(w_embs)
            else:
                chunk_word_embs.append(np.zeros((1, query_word_embs.shape[1])))

        colbert_idx, colbert_scores = select_colbert_style(query_word_embs, chunk_word_embs, n_select)

        # Lumisift
        lumi_idx, lumi_scores = select_lumisift(chunks, evaluator, n_select)

        # Hybrid
        hybrid_idx, hybrid_scores = select_hybrid(emb_scores, lumi_scores, 0.3, n_select)

        # ── Evaluate retention per method ──────────────────────────────

        method_results = {
            "BM25": bm25_idx,
            "ColBERT": colbert_idx,
            "Embedding": emb_idx,
            "Lumisift": lumi_idx,
            "Hybrid (a=0.3)": hybrid_idx,
        }

        for method_name, idx in method_results.items():
            selected_text = " ".join(chunks[j] for j in idx)
            retained = facts_retained(all_facts, selected_text)
            method_stats[method_name]["retained"] += len(retained)
            method_stats[method_name]["total"] += len(all_facts)
            method_stats[method_name]["articles"] += 1

            # Track per fact type
            for fact in all_facts:
                ft = fact["type"]
                if ft not in type_stats:
                    type_stats[ft] = {m: {"kept": 0, "total": 0} for m in methods}
                type_stats[ft][method_name]["total"] += 1
                if fact["value"] in selected_text:
                    type_stats[ft][method_name]["kept"] += 1

        # Per-article winner
        best_count = 0
        for method_name, idx in method_results.items():
            selected_text = " ".join(chunks[j] for j in idx)
            retained = len(facts_retained(all_facts, selected_text))
            if retained > best_count:
                best_count = retained

        for method_name, idx in method_results.items():
            selected_text = " ".join(chunks[j] for j in idx)
            retained = len(facts_retained(all_facts, selected_text))
            if retained == best_count:
                method_stats[method_name]["wins"] += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(articles)} articles...")

    # ─── Results ───────────────────────────────────────────────────────────

    print(f"\n\n{'='*70}")
    print("  RESULTS: 5-Method Numerical Retention Comparison")
    print(f"{'='*70}\n")

    print(f"  Articles processed:    {articles_processed}")
    print(f"  With numerical facts:  {articles_with_facts}")
    total_facts = method_stats["BM25"]["total"]
    print(f"  Total facts tested:    {total_facts}\n")

    print(f"  {'Method':<22} {'Retained':>10} {'Rate':>8} {'vs BM25':>10} {'Wins':>8}")
    print(f"  {'-'*62}")

    bm25_rate = method_stats["BM25"]["retained"] / max(1, method_stats["BM25"]["total"]) * 100

    ranking = []
    for method in methods:
        s = method_stats[method]
        rate = s["retained"] / max(1, s["total"]) * 100
        delta = rate - bm25_rate
        ranking.append((method, s["retained"], s["total"], rate, delta, s["wins"]))

    ranking.sort(key=lambda x: -x[3])

    for method, retained, total, rate, delta, wins in ranking:
        delta_str = f"+{delta:.1f}pp" if delta > 0 else f"{delta:.1f}pp"
        if method == "BM25":
            delta_str = "baseline"
        print(f"  {method:<22} {retained:>7}/{total} {rate:>7.1f}% {delta_str:>10} {wins:>7}")

    # ─── Fact type breakdown ───────────────────────────────────────────────

    print(f"\n  Fact Type Breakdown:\n")
    print(f"  {'Type':<20} {'BM25':>7} {'ColBERT':>8} {'Embed':>7} {'Lumi':>7} {'Hybrid':>7} {'Best':>10}")
    print(f"  {'-'*72}")

    for ft in sorted(type_stats.keys(), key=lambda x: type_stats[x]["BM25"]["total"], reverse=True):
        ts = type_stats[ft]
        rates = {}
        for m in methods:
            t = ts[m]["total"]
            k = ts[m]["kept"]
            rates[m] = k / max(1, t) * 100

        best = max(rates.items(), key=lambda x: x[1])
        best_name = best[0].split(" ")[0]  # Short name

        print(f"  {ft:<20} {rates['BM25']:>6.1f}% {rates['ColBERT']:>7.1f}% "
              f"{rates['Embedding']:>6.1f}% {rates['Lumisift']:>6.1f}% "
              f"{rates['Hybrid (a=0.3)']:>6.1f}% {best_name:>10}")

    # ─── Key Finding ───────────────────────────────────────────────────────

    best_method = ranking[0]
    worst_method = ranking[-1]

    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}\n")
    print(f"  Best method:    {best_method[0]} ({best_method[3]:.1f}%)")
    print(f"  Worst method:   {worst_method[0]} ({worst_method[3]:.1f}%)")
    print(f"  Delta:          +{best_method[3] - worst_method[3]:.1f}pp\n")

    for method, retained, total, rate, delta, wins in ranking:
        if method == "BM25":
            continue
        emoji = ">" if delta > 0 else "<" if delta < 0 else "="
        print(f"  {method:<22} {emoji} BM25 by {abs(delta):>5.1f}pp")

    # ─── Save ──────────────────────────────────────────────────────────────

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "articles_processed": articles_processed,
            "articles_with_facts": articles_with_facts,
            "total_facts": total_facts,
            "selection_ratio": "50% (top_k = len(chunks) // 2)",
            "methods": methods,
        },
        "results": {
            m: {
                "retained": method_stats[m]["retained"],
                "total": method_stats[m]["total"],
                "rate_pct": round(method_stats[m]["retained"] / max(1, method_stats[m]["total"]) * 100, 1),
                "wins": method_stats[m]["wins"],
            }
            for m in methods
        },
        "fact_types": {
            ft: {
                m: round(type_stats[ft][m]["kept"] / max(1, type_stats[ft][m]["total"]) * 100, 1)
                for m in methods
            }
            for ft in type_stats
        },
        "ranking": [r[0] for r in ranking],
    }

    os.makedirs("benchmark_data", exist_ok=True)
    out_path = os.path.join("benchmark_data", "baseline_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
