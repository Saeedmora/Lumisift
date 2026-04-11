"""
Information Loss Taxonomy
==========================
Measures WHAT information types are systematically lost by different
retrieval methods. This is the scientific core: we don't just show
that Lumisift is better -- we characterize the failure modes of
embedding-based retrieval.

Information types measured:
  1. Numerical facts (IC50, fold-change, %, concentrations)
  2. Named entities (proteins, genes, drugs, organisms)
  3. Causal statements ("X causes Y", "X leads to Y")
  4. Uncertainty markers ("may", "preliminary", "suggests")
  5. Methodological details ("using PCR", "by HPLC")
  6. Comparative claims ("better than", "superior to", "X-fold")

For each retrieval method, we measure:
  - What % of each information type survives 50% context compression
  - Which types are most vulnerable
  - Whether the loss pattern is systematic or random
"""

import os
import sys
import json
import re
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService

# ─── Information Type Extractors ────────────────────────────────────────

def extract_numerical(text):
    """Quantitative data: numbers with units, percentages, p-values."""
    patterns = [
        r'\b\d+\.?\d*\s*%',
        r'\b\d+\.?\d*[-\s]?fold\b',
        r'\b(?:IC50|EC50)\s*[=:~]?\s*\d+\.?\d*\s*(?:nM|uM|mM)',
        r'\b(?:Kd|Km|kcat|Vmax)\s*[=:~]?\s*\d+\.?\d*',
        r'\b[Pp]\s*[<>=]\s*0?\.\d+',
        r'\b\d+\.?\d*\s*(?:mM|uM|nM|pM|mg/mL|ng/mL|mg/kg|g/L)',
        r'\b\d+\.?\d*\s*(?:hours?|hrs?|min(?:utes?)?|days?)\b',
    ]
    facts = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            facts.add(m.group().strip())
    return list(facts)


def extract_entities(text):
    """Named entities: proteins, genes, drugs, organisms."""
    patterns = [
        # Gene/protein names (uppercase 2-6 letter codes)
        r'\b[A-Z][A-Z0-9]{1,5}\b',
        # Drug-like names (capitalized, ending in -ib, -ab, -ol, -ide, etc.)
        r'\b[A-Z][a-z]+(?:inib|umab|izumab|olol|azole|mycin|cillin|navir|vir|tide)\b',
        # Organism names (italic-style, Latin binomials)
        r'\b(?:E\.\s*coli|S\.\s*cerevisiae|P\.\s*falciparum|H\.\s*sapiens)\b',
        # Specific technique mentions
        r'\bCRISPR[-/]?Cas[0-9]*\b',
    ]
    entities = set()
    for p in patterns:
        for m in re.finditer(p, text):
            val = m.group().strip()
            # Filter common non-entity uppercase words
            if val not in {"THE", "AND", "FOR", "BUT", "NOT", "THIS", "THAT", "WITH",
                          "FROM", "INTO", "ALSO", "EACH", "BOTH", "BEEN", "WERE",
                          "HAVE", "HAS", "WAS", "ARE", "CAN", "MAY", "WILL",
                          "DNA", "RNA", "PCR", "NMR", "HIV", "ATP", "GTP",
                          "ALL", "NEW", "USE", "TWO", "ONE", "ITS"}:
                entities.add(val)
    return list(entities)


def extract_causal(text):
    """Causal statements: cause-effect relationships."""
    patterns = [
        r'[^.]*\b(?:causes?|caused|causing)\b[^.]*\.',
        r'[^.]*\b(?:leads?\s+to|led\s+to|leading\s+to)\b[^.]*\.',
        r'[^.]*\b(?:results?\s+in|resulted\s+in)\b[^.]*\.',
        r'[^.]*\b(?:induces?|induced|inducing)\b[^.]*\.',
        r'[^.]*\b(?:inhibits?|inhibited|inhibiting)\b[^.]*\.',
        r'[^.]*\b(?:activates?|activated|activating)\b[^.]*\.',
        r'[^.]*\b(?:promotes?|promoted)\b[^.]*\.',
        r'[^.]*\b(?:suppresses?|suppressed)\b[^.]*\.',
        r'[^.]*\b(?:enhances?|enhanced)\b[^.]*\.',
        r'[^.]*\b(?:reduces?|reduced)\b[^.]*\.',
    ]
    statements = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            # Use a short fingerprint (first 50 chars) to identify the statement
            stmt = m.group().strip()[:80]
            statements.add(stmt)
    return list(statements)


def extract_uncertainty(text):
    """Uncertainty markers: hedging language."""
    patterns = [
        r'\b(?:may|might|could|possibly|potentially)\s+\w+',
        r'\b(?:suggests?|suggested|suggesting)\b',
        r'\b(?:preliminary|tentative|putative)\b',
        r'\b(?:hypothesize[sd]?|speculate[sd]?)\b',
        r'\b(?:remains?\s+(?:to be|unclear|unknown))\b',
        r'\b(?:further\s+(?:studies|research|investigation))\b',
        r'\b(?:not\s+(?:fully|completely|entirely)\s+understood)\b',
    ]
    markers = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            markers.add(m.group().strip().lower())
    return list(markers)


def extract_methods(text):
    """Methodological details: techniques and protocols."""
    patterns = [
        r'\b(?:using|via|by)\s+(?:PCR|HPLC|SDS-PAGE|Western\s+blot|ELISA|NMR|X-ray)',
        r'\b(?:performed|conducted|carried\s+out)\s+(?:using|with|by)\b',
        r'\b(?:transfect(?:ed|ion)|electropor(?:at(?:ed|ion)))\b',
        r'\b(?:incubat(?:ed|ion)|centrifug(?:ed|ation))\b',
        r'\b(?:chromatograph(?:y|ic)|spectroscop(?:y|ic))\b',
        r'\b(?:flow\s+cytometry|mass\s+spectrometry)\b',
        r'\b(?:CRISPR|siRNA|shRNA|qPCR|RT-PCR)\b',
        r'\b(?:mutagenesis|crystallography|cryoEM|cryo-EM)\b',
    ]
    methods = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            methods.add(m.group().strip())
    return list(methods)


def extract_comparative(text):
    """Comparative claims: better/worse/superior/inferior."""
    patterns = [
        r'[^.]*\b(?:better|worse|superior|inferior)\s+(?:to|than)\b[^.]*',
        r'[^.]*\b\d+\.?\d*[-\s]?fold\s+(?:higher|lower|greater|increase|decrease|improvement)\b[^.]*',
        r'[^.]*\b(?:significantly|markedly|dramatically)\s+(?:higher|lower|increased|decreased|improved|reduced)\b[^.]*',
        r'[^.]*\b(?:outperform|surpass)\w*\b[^.]*',
    ]
    claims = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            claims.add(m.group().strip()[:80])
    return list(claims)


# ─── Retention Measurement ──────────────────────────────────────────────

def measure_retention(items, selected_text):
    """What fraction of items appear in the selected text?"""
    if not items:
        return None  # No items to measure
    kept = sum(1 for item in items if item in selected_text)
    return kept / len(items)


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  INFORMATION LOSS TAXONOMY")
    print("  What does each retrieval method systematically lose?")
    print("=" * 70)
    print()

    articles_path = os.path.join("benchmark_data", "pubmed_articles.json")
    with open(articles_path, "r", encoding="utf-8") as f:
        all_articles = json.load(f)

    articles = [a for a in all_articles if len(a.get("abstract", "").split()) > 50]
    print(f"Articles: {len(articles)}\n")

    print("Loading models...")
    embedder = EmbeddingService()
    evaluator = SevenAxesEvaluator(use_llm=False)
    print("Ready.\n")

    # Information types
    info_types = {
        "Numerical Facts": extract_numerical,
        "Named Entities": extract_entities,
        "Causal Statements": extract_causal,
        "Uncertainty Markers": extract_uncertainty,
        "Methodological Details": extract_methods,
        "Comparative Claims": extract_comparative,
    }

    methods = ["Embedding", "BM25", "Lumisift", "Hybrid"]

    # Per information type, per method: list of retention rates
    loss_data = {it: {m: [] for m in methods} for it in info_types}
    type_counts = {it: 0 for it in info_types}

    for i, article in enumerate(articles):
        abstract = article.get("abstract", "")
        title = article.get("title", "")[:80]

        # Chunk
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

        n_select = max(1, len(chunks) // 2)
        query = title

        # ── Score with each method ────────────────────────────────────

        # Embedding
        q_emb = embedder.embed(query)
        c_embs = embedder.embed_many(chunks)
        q_n = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        c_n = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-8)
        emb_scores = c_n @ q_n
        emb_idx = np.argsort(emb_scores)[::-1][:n_select]

        # BM25
        tokenized = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_idx = np.argsort(bm25_scores)[::-1][:n_select]

        # Lumisift (specificity-driven)
        lumi_scores = []
        for chunk in chunks:
            axes, _ = evaluator.evaluate(chunk)
            rel = abs(axes.get("relevance", 0))
            risk = abs(axes.get("risk", 0))
            trust = axes.get("trust", 0.5)
            spec = axes.get("specificity", 0.0)
            s_boost = 1.0 + spec * 0.8
            lumi_scores.append(rel * (1 + risk) * (0.5 + trust * 0.5) * s_boost)
        lumi_scores = np.array(lumi_scores)
        lumi_idx = np.argsort(lumi_scores)[::-1][:n_select]

        # Hybrid
        def normalize(s):
            s = np.array(s, dtype=float)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-8)
        hybrid = 0.3 * normalize(emb_scores) + 0.7 * normalize(lumi_scores)
        hybrid_idx = np.argsort(hybrid)[::-1][:n_select]

        method_indices = {
            "Embedding": emb_idx,
            "BM25": bm25_idx,
            "Lumisift": lumi_idx,
            "Hybrid": hybrid_idx,
        }

        # ── Extract information from full abstract ────────────────────
        for type_name, extractor in info_types.items():
            items = extractor(abstract)
            if not items:
                continue
            type_counts[type_name] += len(items)

            for method_name, idx in method_indices.items():
                selected = " ".join(chunks[j] for j in idx)
                retention = measure_retention(items, selected)
                if retention is not None:
                    loss_data[type_name][method_name].append(retention)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(articles)}...")

    # ─── Results ────────────────────────────────────────────────────────

    print(f"\n\n{'='*78}")
    print("  INFORMATION LOSS TAXONOMY: Results")
    print(f"{'='*78}\n")

    # Summary table
    print(f"  {'Information Type':<24} {'Items':>7}  {'Embed':>7} {'BM25':>7} {'Lumi':>7} {'Hybrid':>7}  {'Most Lost By':>14}")
    print(f"  {'-'*82}")

    results = {}
    for type_name in info_types:
        n_items = type_counts[type_name]
        rates = {}
        for m in methods:
            vals = loss_data[type_name][m]
            rates[m] = np.mean(vals) * 100 if vals else 0

        worst = min(rates.items(), key=lambda x: x[1])
        best = max(rates.items(), key=lambda x: x[1])

        print(f"  {type_name:<24} {n_items:>7}  {rates['Embedding']:>6.1f}% {rates['BM25']:>6.1f}% "
              f"{rates['Lumisift']:>6.1f}% {rates['Hybrid']:>6.1f}%  {worst[0]:>14}")

        results[type_name] = {
            "total_items": n_items,
            "n_articles": len(loss_data[type_name]["Embedding"]),
            "retention_pct": {m: round(rates[m], 1) for m in methods},
            "most_lost_by": worst[0],
            "best_retained_by": best[0],
            "loss_delta_pp": round(best[1] - worst[1], 1),
        }

    # ─── Loss Patterns ─────────────────────────────────────────────────

    print(f"\n{'='*78}")
    print("  LOSS PATTERN ANALYSIS")
    print(f"{'='*78}\n")

    print("  Systematic losses (embedding-based retrieval):\n")
    for type_name, data in sorted(results.items(), key=lambda x: x[1]["retention_pct"]["Embedding"]):
        emb_rate = data["retention_pct"]["Embedding"]
        lumi_rate = data["retention_pct"]["Lumisift"]
        gap = lumi_rate - emb_rate
        severity = "SEVERE" if emb_rate < 50 else "MODERATE" if emb_rate < 70 else "MILD"
        recoverable = "YES" if gap > 10 else "PARTIAL" if gap > 3 else "NO"
        print(f"    {type_name:<24} lost={100-emb_rate:.0f}%  severity={severity:<8}  "
              f"recoverable={recoverable}  (Lumi: +{gap:.0f}pp)")

    # ─── Key Findings ──────────────────────────────────────────────────

    print(f"\n{'='*78}")
    print("  KEY FINDINGS")
    print(f"{'='*78}\n")

    # Find which type Lumisift helps most vs least
    lumi_gains = []
    for type_name, data in results.items():
        gain = data["retention_pct"]["Lumisift"] - data["retention_pct"]["Embedding"]
        lumi_gains.append((type_name, gain, data["retention_pct"]["Embedding"], data["retention_pct"]["Lumisift"]))

    lumi_gains.sort(key=lambda x: -x[1])

    print("  Where specificity-based selection helps most:\n")
    for name, gain, emb, lumi in lumi_gains:
        bar = "#" * max(1, int(gain / 2))
        print(f"    {name:<24} Embed={emb:>5.1f}% -> Lumi={lumi:>5.1f}%  (+{gain:.1f}pp)  {bar}")

    # ─── Save ──────────────────────────────────────────────────────────

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "articles": len(articles),
            "purpose": "Characterize systematic information loss patterns across retrieval methods",
            "information_types": list(info_types.keys()),
            "retrieval_methods": methods,
        },
        "results": results,
        "loss_ranking": [
            {"type": name, "embedding_loss_pct": round(100 - gain[2], 1),
             "lumisift_recovery_pp": round(gain[1], 1)}
            for name, gain_val, *_ in lumi_gains
            for gain in [(gain_val, _[0], _[1])]  # unpack
        ] if False else [  # simplified
            {"type": name, "embedding_retention": emb, "lumisift_retention": lumi, "gain_pp": round(gain, 1)}
            for name, gain, emb, lumi in lumi_gains
        ],
    }

    os.makedirs("benchmark_data", exist_ok=True)
    out_path = os.path.join("benchmark_data", "information_loss_taxonomy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
