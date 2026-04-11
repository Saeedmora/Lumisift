"""
NCBI PubMed Benchmark for Lumisift
====================================
Fetches ~500 real scientific abstracts from PubMed across
multiple domains (drug discovery, protein engineering,
protein extraction, directed evolution, enzyme optimization).

Processes through the 8-axis pipeline and produces
scientifically valid benchmark metrics.

Data Source: NCBI PubMed (https://pubmed.ncbi.nlm.nih.gov/)
"""

import os
import sys
import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Step 1: Fetch PubMed Article IDs ─────────────────────────────────────────

# ─── PubMed Search Queries ─────────────────────────────────────────────────────

SEARCH_QUERIES = [
    ("protein engineering AND directed evolution", 150),
    ("drug discovery AND IC50 AND inhibitor", 120),
    ("protein extraction AND purification AND yield", 100),
    ("enzyme optimization AND catalytic activity", 80),
    ("lipid nanoparticle AND mRNA delivery", 80),
]


def search_pubmed(query: str, max_results: int = 100) -> list:
    """Search PubMed and return article IDs."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    print(f"  Searching: '{query}' (max {max_results})...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ids = data["esearchresult"]["idlist"]
    total = data["esearchresult"]["count"]
    print(f"  Found {total} total results, fetching top {len(ids)}")
    return ids


def search_all_queries() -> list:
    """Search PubMed with multiple queries and deduplicate by PMID."""
    print("[1/5] Searching PubMed across 5 domains...")
    all_ids = []
    seen = set()
    for query, max_results in SEARCH_QUERIES:
        ids = search_pubmed(query, max_results)
        new_ids = [pid for pid in ids if pid not in seen]
        seen.update(new_ids)
        all_ids.extend(new_ids)
        print(f"       +{len(new_ids)} new IDs (total unique: {len(all_ids)})")
        time.sleep(0.5)  # NCBI rate limit
    print(f"\n  Total unique article IDs: {len(all_ids)}")
    return all_ids


# ─── Step 2: Fetch Abstracts in Batches ───────────────────────────────────────

def fetch_abstracts(pmids: list, batch_size: int = 20) -> list:
    """Fetch article details from PubMed in batches."""
    articles = []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    print(f"[2/5] Fetching {len(pmids)} abstracts from NCBI...")

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid = article.findtext(".//PMID", "")
                    title = article.findtext(".//ArticleTitle", "")

                    # Get abstract text (may have multiple sections)
                    abstract_parts = []
                    for abs_text in article.findall(".//AbstractText"):
                        label = abs_text.get("Label", "")
                        text = abs_text.text or ""
                        # Handle mixed content with nested tags
                        full_text = ET.tostring(abs_text, encoding="unicode", method="text").strip()
                        if label:
                            abstract_parts.append(f"{label}: {full_text}")
                        else:
                            abstract_parts.append(full_text)

                    abstract = "\n\n".join(abstract_parts)

                    if not abstract or len(abstract) < 100:
                        continue

                    # Get journal and year
                    journal = article.findtext(".//Journal/Title", "")
                    year = article.findtext(".//PubDate/Year", "")
                    if not year:
                        year = article.findtext(".//PubDate/MedlineDate", "")[:4] if article.findtext(".//PubDate/MedlineDate") else ""

                    # Get MeSH terms
                    mesh_terms = [
                        m.findtext("DescriptorName", "")
                        for m in article.findall(".//MeshHeading")
                    ]

                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                        "mesh_terms": mesh_terms[:10],
                        "char_count": len(abstract),
                        "word_count": len(abstract.split()),
                    })
                except Exception as e:
                    continue

            print(f"       Batch {i // batch_size + 1}/{(len(pmids) + batch_size - 1) // batch_size}: "
                  f"got {len(articles)} articles so far")
            time.sleep(0.4)  # NCBI rate limit: max 3 requests/sec

        except Exception as e:
            print(f"       Batch error: {e}")
            time.sleep(1)

    return articles


# ─── Step 3: Process Through Pipeline ─────────────────────────────────────────

def process_articles(articles: list) -> dict:
    """Process articles through the Logical Rooms pipeline."""
    from core.pipeline import LogicalRoomsPipeline
    from core.atom import _count_tokens, AXIS_NAMES

    print(f"\n[3/5] Processing {len(articles)} articles through pipeline...")

    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)

    results = []
    total_original_tokens = 0
    total_compressed_tokens = 0
    total_processing_ms = 0
    all_axes = {axis: [] for axis in AXIS_NAMES}

    for i, article in enumerate(articles):
        start = time.time()

        # Split abstract into semantic chunks
        # PubMed abstracts may be: (a) multi-section (BACKGROUND/METHODS/RESULTS)
        # or (b) single block. For single blocks, split on sentences.
        paragraphs = [p.strip() for p in article["abstract"].split("\n\n") if p.strip() and len(p.strip()) > 20]

        # If only 1 paragraph, split on sentences for finer granularity
        if len(paragraphs) <= 1:
            import re
            text = article["abstract"]
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            # Group into chunks of 2-3 sentences for meaningful atoms
            chunks = []
            buf = []
            for s in sentences:
                buf.append(s.strip())
                if len(buf) >= 2 and len(" ".join(buf)) > 80:
                    chunks.append(" ".join(buf))
                    buf = []
            if buf:
                chunks.append(" ".join(buf))
            paragraphs = [c for c in chunks if len(c) > 30]

        if not paragraphs:
            paragraphs = [article["abstract"]]

        try:
            # Process
            atoms = pipe.process_batch(paragraphs, domain="biotech")
            surfaces = pipe.build_surfaces(atoms, strategy="similarity")

            # Compute metrics
            orig_tokens = sum(a.original_tokens for a in atoms)
            compressed_parts = [a.to_compressed_repr() for a in atoms]
            compressed_text = " ".join(compressed_parts)
            comp_tokens = _count_tokens(compressed_text)

            # Axes-driven selection (pass pre-built atoms to avoid re-processing)
            selection = pipe.select_context(atoms, top_k=max(1, len(atoms) // 2))
            selected_tokens = selection.compressed_tokens if selection else comp_tokens

            elapsed = (time.time() - start) * 1000

            # Collect axes values
            for atom in atoms:
                for axis in AXIS_NAMES:
                    all_axes[axis].append(atom.axes.get(axis, 0))

            result = {
                "pmid": article["pmid"],
                "title": article["title"][:80],
                "paragraphs": len(paragraphs),
                "atoms": len(atoms),
                "surfaces": len(surfaces),
                "original_tokens": orig_tokens,
                "compressed_tokens": comp_tokens,
                "selected_tokens": selected_tokens,
                "compression_ratio": round(1 - comp_tokens / max(1, orig_tokens), 4),
                "selection_ratio": round(1 - selected_tokens / max(1, orig_tokens), 4),
                "processing_ms": round(elapsed, 1),
                "mean_tension": round(float(sum(a.tension for a in atoms) / max(1, len(atoms))), 4),
                "mean_confidence": round(float(sum(a.confidence for a in atoms) / max(1, len(atoms))), 4),
            }
            results.append(result)

            total_original_tokens += orig_tokens
            total_compressed_tokens += comp_tokens
            total_processing_ms += elapsed

            if (i + 1) % 10 == 0:
                print(f"       Processed {i + 1}/{len(articles)} articles...")

        except Exception as e:
            print(f"       Error on article {i}: {e}")
            continue

    return {
        "results": results,
        "total_original_tokens": total_original_tokens,
        "total_compressed_tokens": total_compressed_tokens,
        "total_processing_ms": total_processing_ms,
        "all_axes": all_axes,
    }


# ─── Step 4: Compute Benchmark Statistics ─────────────────────────────────────

def compute_benchmark(data: dict, articles: list) -> dict:
    """Compute comprehensive benchmark statistics."""
    import numpy as np

    results = data["results"]
    if not results:
        return {"error": "No results to benchmark"}

    print(f"\n[4/5] Computing benchmark statistics over {len(results)} articles...")

    # Token metrics
    orig_tokens = [r["original_tokens"] for r in results]
    comp_tokens = [r["compressed_tokens"] for r in results]
    sel_tokens = [r["selected_tokens"] for r in results]
    comp_ratios = [r["compression_ratio"] for r in results]
    sel_ratios = [r["selection_ratio"] for r in results]
    times = [r["processing_ms"] for r in results]
    tensions = [r["mean_tension"] for r in results]
    confidences = [r["mean_confidence"] for r in results]

    # Axes distribution
    axes_stats = {}
    for axis, values in data["all_axes"].items():
        if values:
            axes_stats[axis] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "median": round(float(np.median(values)), 4),
            }

    benchmark = {
        "metadata": {
            "query": "protein engineering AND directed evolution",
            "source": "NCBI PubMed",
            "date": datetime.now().isoformat(),
            "articles_fetched": len(articles),
            "articles_processed": len(results),
            "domain": "biotech",
            "evaluator": "heuristic (keyword-based)",
            "tokenizer": "tiktoken cl100k_base (GPT-4 compatible)",
        },
        "corpus_stats": {
            "total_articles": len(results),
            "total_paragraphs": sum(r["paragraphs"] for r in results),
            "total_atoms": sum(r["atoms"] for r in results),
            "total_surfaces": sum(r["surfaces"] for r in results),
            "total_original_tokens": sum(orig_tokens),
            "avg_abstract_tokens": round(float(np.mean(orig_tokens)), 1),
            "avg_paragraphs_per_article": round(float(np.mean([r["paragraphs"] for r in results])), 1),
            "avg_atoms_per_article": round(float(np.mean([r["atoms"] for r in results])), 1),
        },
        "compression_benchmark": {
            "note": "Compressed representation = structured atom format (axes + metadata + text)",
            "mean_ratio": round(float(np.mean(comp_ratios)), 4),
            "median_ratio": round(float(np.median(comp_ratios)), 4),
            "std_ratio": round(float(np.std(comp_ratios)), 4),
            "best_ratio": round(float(np.max(comp_ratios)), 4),
            "worst_ratio": round(float(np.min(comp_ratios)), 4),
            "total_original_tokens": sum(orig_tokens),
            "total_compressed_tokens": sum(comp_tokens),
            "overall_ratio": round(1 - sum(comp_tokens) / max(1, sum(orig_tokens)), 4),
        },
        "selection_benchmark": {
            "note": "Axes-driven selection = top-k atoms by tension score (raw text, not structured)",
            "mean_ratio": round(float(np.mean(sel_ratios)), 4),
            "median_ratio": round(float(np.median(sel_ratios)), 4),
            "std_ratio": round(float(np.std(sel_ratios)), 4),
            "best_ratio": round(float(np.max(sel_ratios)), 4),
            "worst_ratio": round(float(np.min(sel_ratios)), 4),
            "total_selected_tokens": sum(sel_tokens),
            "overall_ratio": round(1 - sum(sel_tokens) / max(1, sum(orig_tokens)), 4),
        },
        "performance": {
            "total_processing_ms": round(data["total_processing_ms"], 1),
            "avg_per_article_ms": round(float(np.mean(times)), 1),
            "median_per_article_ms": round(float(np.median(times)), 1),
            "p95_per_article_ms": round(float(np.percentile(times, 95)), 1),
            "throughput_articles_per_sec": round(len(results) / (data["total_processing_ms"] / 1000), 2),
        },
        "axes_distribution": axes_stats,
        "quality_indicators": {
            "mean_tension": round(float(np.mean(tensions)), 4),
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "tension_std": round(float(np.std(tensions)), 4),
            "confidence_std": round(float(np.std(confidences)), 4),
        },
    }

    return benchmark


# ─── Step 5: Export Training Data ─────────────────────────────────────────────

def export_training_data(articles: list, results: list, output_dir: str = "benchmark_data"):
    """Export JSONL training data for fine-tuning."""
    from core.pipeline import LogicalRoomsPipeline
    from core.atom import AXIS_NAMES

    os.makedirs(output_dir, exist_ok=True)

    # Save raw articles
    articles_path = os.path.join(output_dir, "pubmed_articles.json")
    with open(articles_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"       Saved {len(articles)} articles to {articles_path}")

    # Save training JSONL (for fine-tuning axis evaluation)
    training_path = os.path.join(output_dir, "training_data.jsonl")
    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)
    count = 0

    with open(training_path, "w", encoding="utf-8") as f:
        for article in articles:
            import re
            paragraphs = [p.strip() for p in article["abstract"].split("\n\n") if p.strip() and len(p.strip()) > 20]
            if len(paragraphs) <= 1:
                text = article["abstract"]
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
                chunks = []
                buf = []
                for s in sentences:
                    buf.append(s.strip())
                    if len(buf) >= 2 and len(" ".join(buf)) > 80:
                        chunks.append(" ".join(buf))
                        buf = []
                if buf:
                    chunks.append(" ".join(buf))
                paragraphs = [c for c in chunks if len(c) > 30]
            if not paragraphs:
                paragraphs = [article["abstract"]]

            try:
                atoms = pipe.process_batch(paragraphs, domain="biotech")
                for atom in atoms:
                    entry = {
                        "text": atom.text,
                        "axes": {k: round(v, 4) for k, v in atom.axes.items()},
                        "domain": "biotech",
                        "category": atom.ontology_category.value if atom.ontology_category else "unknown",
                        "tension": round(atom.tension, 4),
                        "confidence": round(atom.confidence, 4),
                        "pmid": article["pmid"],
                        "source": "pubmed",
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1
            except Exception:
                continue

    print(f"       Saved {count} training samples to {training_path}")
    return count


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Lumisift -- PubMed Benchmark")
    print("  Drug Discovery, Protein Engineering & Extraction")
    print("=" * 70)
    print()

    # Step 1: Search across 5 domains
    pmids = search_all_queries()

    # Step 2: Fetch
    articles = fetch_abstracts(pmids)
    print(f"\n       Total articles with abstracts: {len(articles)}")

    # Step 3: Process
    data = process_articles(articles)

    # Step 4: Benchmark
    benchmark = compute_benchmark(data, articles)

    # Step 5: Export
    output_dir = "benchmark_data"
    os.makedirs(output_dir, exist_ok=True)

    training_count = export_training_data(articles, data["results"], output_dir)

    # Save benchmark results
    benchmark_path = os.path.join(output_dir, "benchmark_results.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    # ─── Print Results ────────────────────────────────────────────────────

    print("\n")
    print("=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    meta = benchmark["metadata"]
    print(f"\n  Source:    {meta['source']}")
    print(f"  Query:    {meta['query']}")
    print(f"  Articles: {meta['articles_processed']} processed")
    print(f"  Domain:   {meta['domain']}")
    print(f"  Tokenizer: {meta['tokenizer']}")

    cs = benchmark["corpus_stats"]
    print(f"\n  Corpus:")
    print(f"    Total atoms:      {cs['total_atoms']}")
    print(f"    Total tokens:     {cs['total_original_tokens']:,}")
    print(f"    Avg tokens/article: {cs['avg_abstract_tokens']}")
    print(f"    Avg atoms/article:  {cs['avg_atoms_per_article']}")

    cb = benchmark["compression_benchmark"]
    print(f"\n  Compression (structured representation):")
    print(f"    Overall ratio:    {cb['overall_ratio']:.1%}")
    print(f"    Mean per-article: {cb['mean_ratio']:.1%}")
    print(f"    Median:           {cb['median_ratio']:.1%}")
    print(f"    Best:             {cb['best_ratio']:.1%}")
    print(f"    Worst:            {cb['worst_ratio']:.1%}")

    sb = benchmark["selection_benchmark"]
    print(f"\n  Selection (axes-driven top-k):")
    print(f"    Overall ratio:    {sb['overall_ratio']:.1%}")
    print(f"    Mean per-article: {sb['mean_ratio']:.1%}")
    print(f"    Median:           {sb['median_ratio']:.1%}")
    print(f"    Best:             {sb['best_ratio']:.1%}")
    print(f"    Worst:            {sb['worst_ratio']:.1%}")

    perf = benchmark["performance"]
    print(f"\n  Performance:")
    print(f"    Total time:       {perf['total_processing_ms'] / 1000:.1f}s")
    print(f"    Avg per article:  {perf['avg_per_article_ms']:.0f}ms")
    print(f"    P95 per article:  {perf['p95_per_article_ms']:.0f}ms")
    print(f"    Throughput:       {perf['throughput_articles_per_sec']:.1f} articles/sec")

    print(f"\n  Axes Distribution (across {cs['total_atoms']} atoms):")
    for axis, stats in benchmark["axes_distribution"].items():
        print(f"    {axis:12s}: mean={stats['mean']:+.3f}  std={stats['std']:.3f}  range=[{stats['min']:+.3f}, {stats['max']:+.3f}]")

    qi = benchmark["quality_indicators"]
    print(f"\n  Quality:")
    print(f"    Mean tension:     {qi['mean_tension']:.4f}")
    print(f"    Mean confidence:  {qi['mean_confidence']:.4f}")

    print(f"\n  Training data:  {training_count} samples exported")
    print(f"  Output:         {output_dir}/")
    print(f"    - pubmed_articles.json   (raw articles)")
    print(f"    - training_data.jsonl    (fine-tuning samples)")
    print(f"    - benchmark_results.json (this report)")

    print("\n" + "=" * 70)
    print("  Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
