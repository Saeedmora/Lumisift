"""
PubMedQA Benchmark -- Lumisift vs Full-Context RAG
====================================================
Uses the official PubMedQA dataset format to test: can Lumisift
select 50% of context and still answer yes/no/maybe correctly?

Methodology:
  1. Load PubMedQA-style questions from our 95-article corpus
  2. Generate yes/no/maybe questions about each article (1 Gemini call)
  3. Answer with full text and Lumisift-selected text (2 Gemini calls)
  4. Compare correctness

This is a standardized benchmark format recognized by the research community.
"""

import os
import sys
import json
import re
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from google import genai
from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-3-flash-preview"
client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except Exception as e:
            wait = min(60, (attempt + 1) * 10)
            print(f"  API error (attempt {attempt+1}): {e}")
            print(f"  Waiting {wait}s...")
            time.sleep(wait)
    return ""


def select_by_similarity(query: str, chunks: list, embedder: EmbeddingService, top_k: int) -> list:
    """Standard RAG: cosine similarity selection."""
    query_emb = embedder.embed(query)
    chunk_embs = embedder.embed_many(chunks)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    sims = chunk_norms @ query_norm
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def select_by_lumisift(chunks: list, evaluator: SevenAxesEvaluator, top_k: int) -> list:
    """Lumisift: multi-axis with specificity boost."""
    scored = []
    for chunk in chunks:
        axes, cat = evaluator.evaluate(chunk)
        rel = abs(axes.get("relevance", 0))
        risk = abs(axes.get("risk", 0))
        trust = axes.get("trust", 0.5)
        spec = axes.get("specificity", 0.0)
        s_boost = 1.0 + spec * 0.8
        score = rel * (1 + risk) * (0.5 + trust * 0.5) * s_boost
        scored.append({"text": chunk, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s["text"] for s in scored[:top_k]]


def main():
    print("=" * 70)
    print("  PUBMEDQA-STYLE BENCHMARK")
    print("  Lumisift vs Embedding Similarity vs Full Context")
    print("=" * 70)
    print()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY required. Set in .env file.")
        sys.exit(1)

    # Load articles
    articles_path = os.path.join("benchmark_data", "pubmed_articles.json")
    with open(articles_path, "r", encoding="utf-8") as f:
        all_articles = json.load(f)

    # 15 articles with substantial content (manageable for API calls)
    articles = [a for a in all_articles if len(a.get("abstract", "").split()) > 100][:15]
    print(f"Selected {len(articles)} articles for PubMedQA benchmark\n")

    # Initialize
    print("Loading models...")
    embedder = EmbeddingService()
    evaluator = SevenAxesEvaluator(use_llm=False)
    print("Ready.\n")

    # ─── Step 1: Generate PubMedQA-style questions ─────────────────────────

    print("Step 1: Generating yes/no/maybe questions (1 API call)...")

    articles_block = ""
    for i, article in enumerate(articles):
        articles_block += f"\n---ARTICLE {i}---\nPMID: {article.get('pmid','?')}\nTitle: {article.get('title','')}\nAbstract: {article.get('abstract','')[:500]}\n"

    prompt_questions = f"""You are a biomedical research expert. For each article below, generate ONE yes/no/maybe question
that can be answered from the abstract. The question should test comprehension of a KEY FINDING.

Return a JSON array with exactly {len(articles)} objects.
Each object must have: "idx" (int), "question" (string), "correct_answer" ("yes", "no", or "maybe")

Articles:
{articles_block}

Return ONLY the JSON array, no markdown formatting, no code fences."""

    response = ask_gemini(prompt_questions)
    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)
        questions = json.loads(clean)
        print(f"  Generated {len(questions)} questions.\n")
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw: {response[:500]}")
        sys.exit(1)

    # ─── Step 2: Process articles and prepare contexts ─────────────────────

    print("Step 2: Processing articles and selecting context...")

    eval_data = []
    for i, article in enumerate(articles):
        abstract = article.get("abstract", "")
        title = article.get("title", "")[:60]

        q = next((qa for qa in questions if qa.get("idx") == i), None)
        if not q:
            continue

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

        n_select = max(1, len(chunks) // 2)

        # Method A: Full context
        full_text = abstract

        # Method B: Embedding similarity
        sim_selected = select_by_similarity(q["question"], chunks, embedder, n_select)
        sim_text = " ".join(sim_selected)

        # Method C: Lumisift
        lumi_selected = select_by_lumisift(chunks, evaluator, n_select)
        lumi_text = " ".join(lumi_selected)

        eval_data.append({
            "idx": i,
            "pmid": article.get("pmid", "?"),
            "title": title,
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "full_text": full_text,
            "sim_text": sim_text,
            "lumi_text": lumi_text,
            "n_chunks": len(chunks),
            "n_selected": n_select,
        })
        print(f"  [{i+1}] {title}... ({len(chunks)} chunks, top {n_select})")

    print(f"\n{len(eval_data)} articles ready.\n")

    # ─── Step 3: Answer questions with all 3 methods ───────────────────────

    print("Step 3: Answering questions with 3 methods (1 API call)...")

    answer_block = ""
    for d in eval_data:
        answer_block += f"""
---ARTICLE {d['idx']}---
Question: {d['question']}

FULL CONTEXT: {d['full_text'][:600]}

SIMILARITY-SELECTED CONTEXT ({d['n_selected']}/{d['n_chunks']} chunks): {d['sim_text'][:400]}

LUMISIFT-SELECTED CONTEXT ({d['n_selected']}/{d['n_chunks']} chunks): {d['lumi_text'][:400]}
"""

    prompt_answer = f"""You are a biomedical expert. For each article below, answer the yes/no/maybe question
using THREE different contexts: FULL, SIMILARITY-SELECTED, and LUMISIFT-SELECTED.

For each context, answer ONLY "yes", "no", or "maybe" based on what the context supports.
If the context lacks enough information, answer "maybe".

Return a JSON array with one object per article.
Each object must have: "idx", "answer_full", "answer_similarity", "answer_lumisift"

{answer_block}

Return ONLY the JSON array, no markdown formatting, no code fences."""

    time.sleep(5)
    response_answers = ask_gemini(prompt_answer)

    try:
        clean = response_answers.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)
        answers = json.loads(clean)
        print(f"  Got {len(answers)} answer sets.\n")
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw: {response_answers[:500]}")
        answers = []

    # ─── Step 4: Score ─────────────────────────────────────────────────────

    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    full_correct = 0
    sim_correct = 0
    lumi_correct = 0
    total = 0

    results = []
    for d in eval_data:
        a = next((ans for ans in answers if ans.get("idx") == d["idx"]), None)
        if not a:
            continue

        total += 1
        correct = d["correct_answer"].lower().strip()
        af = a.get("answer_full", "").lower().strip()
        asim = a.get("answer_similarity", "").lower().strip()
        alumi = a.get("answer_lumisift", "").lower().strip()

        fc = 1 if af == correct else 0
        sc = 1 if asim == correct else 0
        lc = 1 if alumi == correct else 0

        full_correct += fc
        sim_correct += sc
        lumi_correct += lc

        results.append({
            "pmid": d["pmid"],
            "title": d["title"],
            "question": d["question"],
            "correct": correct,
            "full": af, "full_correct": bool(fc),
            "similarity": asim, "similarity_correct": bool(sc),
            "lumisift": alumi, "lumisift_correct": bool(lc),
        })

        mark_f = "+" if fc else "X"
        mark_s = "+" if sc else "X"
        mark_l = "+" if lc else "X"
        print(f"  [{d['idx']+1}] {d['title']}...")
        print(f"       Q: {d['question'][:70]}...")
        print(f"       Correct: {correct} | Full[{mark_f}]={af} | Sim[{mark_s}]={asim} | Lumi[{mark_l}]={alumi}")
        print()

    if total > 0:
        full_acc = full_correct / total * 100
        sim_acc = sim_correct / total * 100
        lumi_acc = lumi_correct / total * 100

        print(f"  {'Method':<35} {'Correct':>8} {'Accuracy':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Full Context (100% tokens)':<35} {full_correct:>8}/{total} {full_acc:>9.1f}%")
        print(f"  {'Embedding Similarity (50% tokens)':<35} {sim_correct:>8}/{total} {sim_acc:>9.1f}%")
        print(f"  {'Lumisift (50% tokens)':<35} {lumi_correct:>8}/{total} {lumi_acc:>9.1f}%")
        print()

        if lumi_acc >= full_acc:
            print(f"  KEY: Lumisift matches or beats full context with 50% fewer tokens!")
        elif lumi_acc > sim_acc:
            print(f"  KEY: Lumisift outperforms embedding similarity (+{lumi_acc-sim_acc:.1f}pp)")

        # Save
        output = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "format": "PubMedQA-style yes/no/maybe",
                "articles": total,
                "selection_ratio": "50%",
            },
            "accuracy": {
                "full_context": {"correct": full_correct, "total": total, "pct": round(full_acc, 1)},
                "embedding_similarity": {"correct": sim_correct, "total": total, "pct": round(sim_acc, 1)},
                "lumisift": {"correct": lumi_correct, "total": total, "pct": round(lumi_acc, 1)},
            },
            "per_article": results,
        }

        os.makedirs("benchmark_data", exist_ok=True)
        with open(os.path.join("benchmark_data", "pubmedqa_benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to benchmark_data/pubmedqa_benchmark.json")


if __name__ == "__main__":
    main()
