"""
Ablation Study — Which Axis Matters Most?
==========================================
Systematically removes each of the 8 axes one at a time and measures
the impact on numerical retention. This is essential for credibility:
if removing an axis doesn't change the result, it's not contributing.

Also tests: specificity boost removal, trust weighting removal,
risk amplification removal, and temporal boost removal.

Output: per-axis drop in retention rate, proving each component's contribution.
"""

import os
import sys
import json
import re
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService

# ─── Numerical Fact Extraction ──────────────────────────────────────────

NUMERICAL_PATTERNS = [
    (r'\b\d+\.?\d*\s*%', "percentage"),
    (r'\b\d+\.?\d*[-\s]?fold\b', "fold_change"),
    (r'\b\d+\.?\d*\s*[xX]\b', "fold_change"),
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


# ─── Scoring Variants ──────────────────────────────────────────────────

def score_full(axes):
    """Full Lumisift scoring formula."""
    rel = abs(axes.get("relevance", 0))
    risk = abs(axes.get("risk", 0))
    trust = axes.get("trust", 0.5)
    spec = axes.get("specificity", 0.0)
    s_boost = 1.0 + spec * 0.8
    return rel * (1 + risk) * (0.5 + trust * 0.5) * s_boost


def score_without_axis(axes, removed_axis):
    """Score with one axis zeroed out."""
    modified = dict(axes)
    if removed_axis == "specificity":
        modified["specificity"] = 0.0  # No specificity boost
    elif removed_axis == "trust":
        modified["trust"] = 0.5  # Neutral trust (no influence)
    elif removed_axis == "risk":
        modified["risk"] = 0.0  # No risk amplification
    elif removed_axis == "relevance":
        modified["relevance"] = 0.5  # Neutral relevance
    elif removed_axis == "causality":
        modified["causality"] = 0.0  # No causality signal
    elif removed_axis == "temporal":
        modified["temporal"] = 0.0  # No temporal boost
    elif removed_axis == "ontology":
        modified["ontology"] = 0.0  # No domain signal
    elif removed_axis == "visibility":
        modified["visibility"] = 0.5  # Neutral visibility
    return score_full(modified)


def score_no_specificity_boost(axes):
    """Full formula but specificity boost is always 1.0."""
    rel = abs(axes.get("relevance", 0))
    risk = abs(axes.get("risk", 0))
    trust = axes.get("trust", 0.5)
    return rel * (1 + risk) * (0.5 + trust * 0.5)  # No s_boost


def score_only_specificity(axes):
    """Only specificity, nothing else."""
    return axes.get("specificity", 0.0)


def score_only_relevance(axes):
    """Only relevance, nothing else."""
    return abs(axes.get("relevance", 0))


# ─── Main ──────────────────────────────────────────────────────────────

def run_ablation(articles, evaluator, score_fn, label):
    """Run numerical retention with a given scoring function."""
    total_facts = 0
    retained_facts = 0
    per_article = []

    for article in articles:
        abstract = article.get("abstract", "")
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

        facts = extract_facts(abstract)
        if not facts:
            continue

        n_select = max(1, len(chunks) // 2)

        # Score chunks
        scores = []
        for chunk in chunks:
            axes, _ = evaluator.evaluate(chunk)
            scores.append(score_fn(axes))

        top_idx = np.argsort(scores)[::-1][:n_select]
        selected_text = " ".join(chunks[j] for j in top_idx)

        kept = facts_retained(facts, selected_text)
        total_facts += len(facts)
        retained_facts += len(kept)
        per_article.append(len(kept) / len(facts))

    rate = retained_facts / max(1, total_facts) * 100
    mean = np.mean(per_article) * 100 if per_article else 0
    std = np.std(per_article) * 100 if per_article else 0
    ci95 = 1.96 * std / np.sqrt(len(per_article)) if per_article else 0

    return {
        "label": label,
        "retained": retained_facts,
        "total": total_facts,
        "rate": round(rate, 1),
        "mean_per_article": round(mean, 1),
        "std": round(std, 1),
        "ci95": round(ci95, 1),
        "n_articles": len(per_article),
    }


def main():
    print("=" * 70)
    print("  ABLATION STUDY")
    print("  Which axis drives which result?")
    print("=" * 70)
    print()

    # Load articles
    articles_path = os.path.join("benchmark_data", "pubmed_articles.json")
    with open(articles_path, "r", encoding="utf-8") as f:
        all_articles = json.load(f)

    articles = [a for a in all_articles if len(a.get("abstract", "").split()) > 50]
    print(f"Articles: {len(articles)}\n")

    # Init
    print("Loading evaluator...")
    evaluator = SevenAxesEvaluator(use_llm=False)
    print("Ready.\n")

    # ─── Run all variants ─────────────────────────────────────────────

    results = []

    # 1. Full system (baseline)
    print("  [1/12] Full Lumisift (baseline)...")
    results.append(run_ablation(articles, evaluator, score_full, "Full Lumisift"))

    baseline_rate = results[0]["rate"]

    # 2-9. Remove each axis
    axes_to_remove = [
        "specificity", "relevance", "trust", "risk",
        "causality", "temporal", "ontology", "visibility"
    ]

    for i, axis in enumerate(axes_to_remove):
        print(f"  [{i+2}/12] Without {axis}...")
        fn = lambda axes, ax=axis: score_without_axis(axes, ax)
        results.append(run_ablation(articles, evaluator, fn, f"Without {axis}"))

    # 10. No specificity BOOST (axis exists, boost disabled)
    print("  [10/12] No specificity boost (s_boost = 1.0)...")
    results.append(run_ablation(articles, evaluator, score_no_specificity_boost,
                                "No specificity boost"))

    # 11. Only specificity (nothing else)
    print("  [11/12] Only specificity...")
    results.append(run_ablation(articles, evaluator, score_only_specificity,
                                "Only specificity"))

    # 12. Only relevance (nothing else)
    print("  [12/12] Only relevance...")
    results.append(run_ablation(articles, evaluator, score_only_relevance,
                                "Only relevance"))

    # ─── Results ──────────────────────────────────────────────────────

    print(f"\n\n{'='*74}")
    print("  ABLATION RESULTS")
    print(f"{'='*74}\n")

    print(f"  {'Configuration':<28} {'Rate':>7} {'Delta':>8} {'CI 95%':>9} {'n':>6}")
    print(f"  {'-'*62}")

    for r in results:
        delta = r["rate"] - baseline_rate
        delta_str = f"{delta:+.1f}pp" if r["label"] != "Full Lumisift" else "baseline"
        ci_str = f"+/-{r['ci95']:.1f}pp"
        print(f"  {r['label']:<28} {r['rate']:>6.1f}% {delta_str:>8} {ci_str:>9} {r['n_articles']:>5}")

    # ─── Key findings ─────────────────────────────────────────────────

    print(f"\n{'='*74}")
    print("  KEY FINDINGS")
    print(f"{'='*74}\n")

    # Sort ablations by impact
    ablations = [(r["label"], r["rate"] - baseline_rate) for r in results[1:]]
    ablations.sort(key=lambda x: x[1])  # Most negative = most important

    print("  Most impactful axes (removal causes biggest drop):\n")
    for label, delta in ablations:
        impact = "CRITICAL" if delta < -10 else "SIGNIFICANT" if delta < -3 else "MODERATE" if delta < -1 else "MINIMAL"
        bar = "#" * max(1, int(abs(delta)))
        print(f"    {label:<28} {delta:>+6.1f}pp  [{impact:<12}] {bar}")

    # ─── Save ────────────────────────────────────────────────────────

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "articles": len(articles),
            "methodology": "Remove each axis one at a time, measure numerical retention. "
                           "Also test: no specificity boost, only specificity, only relevance.",
        },
        "baseline": results[0],
        "ablations": results[1:],
        "ranking": [(l, round(d, 1)) for l, d in ablations],
    }

    os.makedirs("benchmark_data", exist_ok=True)
    out_path = os.path.join("benchmark_data", "ablation_study.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
