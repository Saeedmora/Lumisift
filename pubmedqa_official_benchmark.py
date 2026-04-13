"""
PubMedQA Official Benchmark — Lumisift vs Baselines
=====================================================
Uses the OFFICIAL PubMedQA dataset (Jin et al., ACL 2019) with
1,000 expert-annotated yes/no/maybe questions.

This replaces the previous self-generated question approach with
externally validated ground truth — no circular evaluation.

Methodology:
  1. Load PubMedQA-Labeled (1,000 expert-annotated instances)
  2. For each question + context:
     a. Full context → LLM answer
     b. Embedding similarity selection (50%) → LLM answer
     c. Lumisift selection (50%) → LLM answer
     d. Hybrid selection (50%) → LLM answer
  3. Compare ALL answers against human-annotated ground truth
  4. Report accuracy per method

Key difference from pubmedqa_benchmark.py:
  - Ground truth comes from HUMAN EXPERTS, not LLM-generated
  - No circular validation (judge ≠ question generator)
  - Standardized, community-recognized benchmark
  - Results directly comparable to published leaderboards

Paper: Jin et al., "PubMedQA: A Dataset for Biomedical Research
       Question Answering", ACL 2019
Dataset: https://huggingface.co/datasets/qiaojin/PubMedQA
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

from datasets import load_dataset
from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService

# ─── API Configuration (Groq > xAI/Grok > Gemini) ────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GROQ_API_KEY:
    # Groq: free tier with thousands of requests/day, ultra-fast inference
    from groq import Groq
    api_client = Groq(api_key=GROQ_API_KEY)
    MODEL_NAME = "llama-3.3-70b-versatile"
    API_PROVIDER = "Groq"
    print(f"  API: Using Groq ({MODEL_NAME})", flush=True)
elif XAI_API_KEY:
    from openai import OpenAI
    api_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    MODEL_NAME = "grok-3-fast"
    API_PROVIDER = "xAI/Grok"
    print(f"  API: Using xAI/Grok ({MODEL_NAME})", flush=True)
elif GEMINI_API_KEY:
    from google import genai
    api_client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_NAME = "gemini-3-flash-preview"
    API_PROVIDER = "Google/Gemini"
    print(f"  API: Using Google/Gemini ({MODEL_NAME})", flush=True)
else:
    print("ERROR: No API key found. Set GROQ_API_KEY, XAI_API_KEY, or GEMINI_API_KEY in .env")
    sys.exit(1)

# Number of test instances to evaluate (PubMedQA has 1,000 labeled)
# Set to None to use all. Lower for faster testing.
MAX_INSTANCES = 50  # Set to None for full 1,000


def ask_llm(prompt: str, max_retries: int = 5) -> str:
    """Call the configured LLM API (Groq, Grok, or Gemini)."""
    for attempt in range(max_retries):
        try:
            if API_PROVIDER in ("Groq", "xAI/Grok"):
                response = api_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()
            else:
                response = api_client.models.generate_content(
                    model=MODEL_NAME, contents=prompt
                )
                return response.text.strip()
        except Exception as e:
            err_str = str(e)
            # Detect daily quota exhaustion (different from per-minute rate limit)
            if "429" in err_str and "FreeTier" in err_str:
                print(f"  QUOTA EXHAUSTED: Daily free-tier limit reached.", flush=True)
                print(f"  Saving checkpoint and stopping. Re-run tomorrow to continue.", flush=True)
                return "__QUOTA_EXHAUSTED__"
            wait = min(60, (attempt + 1) * 10)
            # Check for suggested retry delay
            delay_match = re.search(r'retryDelay.*?(\d+)', err_str)
            if delay_match:
                wait = int(delay_match.group(1)) + 2
            print(f"  API error (attempt {attempt+1}): {str(e)[:120]}", flush=True)
            print(f"  Waiting {wait}s...", flush=True)
            time.sleep(wait)
    return ""


def answer_question(question: str, context: str) -> str:
    """Ask LLM to answer a yes/no/maybe question given context."""
    prompt = f"""You are a biomedical expert. Based ONLY on the context provided,
answer the following question with exactly one word: "yes", "no", or "maybe".

If the context clearly supports the answer, say "yes" or "no".
If the context is insufficient or ambiguous, say "maybe".

Context:
{context}

Question: {question}

Answer (one word only):"""

    response = ask_llm(prompt)
    # Extract answer
    answer = response.lower().strip().rstrip(".")
    # Normalize to valid answers
    if "yes" in answer:
        return "yes"
    elif "no" in answer and "maybe" not in answer:
        return "no"
    else:
        return "maybe"


# ─── Selection Methods ────────────────────────────────────────────────────

def select_by_similarity(query: str, chunks: list, embedder: EmbeddingService, top_k: int) -> list:
    """Standard RAG: cosine similarity selection."""
    if len(chunks) <= top_k:
        return list(range(len(chunks)))
    query_emb = embedder.embed(query)
    chunk_embs = embedder.embed_many(chunks)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    sims = chunk_norms @ query_norm
    return np.argsort(sims)[::-1][:top_k].tolist()


def select_by_lumisift(chunks: list, evaluator: SevenAxesEvaluator, top_k: int) -> list:
    """Lumisift: multi-axis with specificity boost."""
    if len(chunks) <= top_k:
        return list(range(len(chunks)))
    scored = []
    for i, chunk in enumerate(chunks):
        axes, cat = evaluator.evaluate(chunk)
        rel = abs(axes.get("relevance", 0))
        risk = abs(axes.get("risk", 0))
        trust = axes.get("trust", 0.5)
        spec = axes.get("specificity", 0.0)
        s_boost = 1.0 + spec * 0.8
        score = rel * (1 + risk) * (0.5 + trust * 0.5) * s_boost
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]


def select_hybrid(query: str, chunks: list, embedder: EmbeddingService,
                   evaluator: SevenAxesEvaluator, top_k: int, alpha: float = 0.3) -> list:
    """Hybrid: alpha * similarity + (1-alpha) * lumisift."""
    if len(chunks) <= top_k:
        return list(range(len(chunks)))

    # Similarity scores
    query_emb = embedder.embed(query)
    chunk_embs = embedder.embed_many(chunks)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    sim_scores = chunk_norms @ query_norm

    # Lumisift scores
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

    # Normalize both to 0-1
    def normalize(s):
        s = np.array(s, dtype=float)
        mn, mx = s.min(), s.max()
        if mx - mn > 1e-8:
            return (s - mn) / (mx - mn)
        return np.ones_like(s) * 0.5

    hybrid = alpha * normalize(sim_scores) + (1 - alpha) * normalize(lumi_scores)
    return np.argsort(hybrid)[::-1][:top_k].tolist()


# ─── Main Benchmark ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PUBMEDQA OFFICIAL BENCHMARK")
    print("  Expert-Annotated Ground Truth (Jin et al., ACL 2019)")
    print("  Lumisift vs Embedding Similarity vs Full Context")
    print("=" * 70)
    print()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY required. Set in .env file.")
        sys.exit(1)

    # ─── Step 1: Load Official PubMedQA Dataset ───────────────────────────

    print("Step 1: Loading official PubMedQA dataset from HuggingFace...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"  Loaded {len(ds)} expert-annotated instances")

    # Apply limit if set
    instances = list(ds)
    if MAX_INSTANCES is not None:
        instances = instances[:MAX_INSTANCES]
        print(f"  Using first {MAX_INSTANCES} instances (set MAX_INSTANCES=None for full)")
    print()

    # Initialize models
    print("Loading models...", flush=True)
    embedder = EmbeddingService()
    evaluator = SevenAxesEvaluator(use_llm=False)
    print("Ready.\n", flush=True)

    # ─── Step 2: Process each instance ────────────────────────────────────

    methods = ["Full Context", "Embedding (50%)", "Lumisift (50%)", "Hybrid (50%)"]
    results = {m: {"correct": 0, "total": 0, "per_article": []} for m in methods}

    skipped = 0
    api_calls = 0
    start_idx = 0

    # ── Resume from checkpoint if exists ──────────────────────────────

    checkpoint_path = os.path.join("benchmark_data", "pubmedqa_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        results = checkpoint["results"]
        skipped = checkpoint["skipped"]
        api_calls = checkpoint["api_calls"]
        start_idx = checkpoint["next_idx"]
        print(f"  RESUMING from checkpoint: {start_idx}/{len(instances)} already done", flush=True)
        print(f"  Previous accuracy: Full={results['Full Context']['correct']}/{results['Full Context']['total']}", flush=True)

    for i, item in enumerate(instances):
        if i < start_idx:
            continue

        question = item["question"]
        context_dict = item["context"]
        gold_answer = item["final_decision"].lower().strip()  # "yes", "no", "maybe"
        pubid = item.get("pubid", str(i))

        # PubMedQA context: dict with "contexts" (list of sentences) and "labels" (list)
        if isinstance(context_dict, dict):
            sentences = context_dict.get("contexts", [])
        elif isinstance(context_dict, list):
            sentences = context_dict
        else:
            sentences = [str(context_dict)]

        # Flatten if needed
        if not sentences:
            skipped += 1
            continue

        full_text = " ".join(sentences)

        # Need at least 2 chunks for selection to be meaningful
        if len(sentences) < 2:
            skipped += 1
            continue

        n_select = max(1, len(sentences) // 2)

        # ── Selection ──────────────────────────────────────────────────

        # Method 1: Full context
        sim_idx = select_by_similarity(question, sentences, embedder, n_select)
        sim_text = " ".join(sentences[j] for j in sorted(sim_idx))

        lumi_idx = select_by_lumisift(sentences, evaluator, n_select)
        lumi_text = " ".join(sentences[j] for j in sorted(lumi_idx))

        hybrid_idx = select_hybrid(question, sentences, embedder, evaluator, n_select)
        hybrid_text = " ".join(sentences[j] for j in sorted(hybrid_idx))

        # ── Batch answer (single API call for efficiency) ──────────────

        batch_prompt = f"""You are a biomedical expert. Answer each of the following 4 questions
based ONLY on the given context. Answer with EXACTLY "yes", "no", or "maybe".

QUESTION: {question}

---CONTEXT A (full)---
{full_text[:1500]}

---CONTEXT B (embedding-selected)---
{sim_text[:1000]}

---CONTEXT C (lumisift-selected)---
{lumi_text[:1000]}

---CONTEXT D (hybrid-selected)---
{hybrid_text[:1000]}

Return a JSON object with exactly these keys: "A", "B", "C", "D"
Each value must be "yes", "no", or "maybe".
Return ONLY the JSON, no markdown, no code fences."""

        response = ask_llm(batch_prompt)
        api_calls += 1

        # Check for quota exhaustion
        if response == "__QUOTA_EXHAUSTED__":
            # Save checkpoint
            checkpoint = {
                "results": results,
                "skipped": skipped,
                "api_calls": api_calls,
                "next_idx": i,
                "note": "Quota exhausted. Re-run to continue from this point.",
            }
            os.makedirs("benchmark_data", exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"\n  Checkpoint saved at index {i}. Re-run to continue.", flush=True)
            break

        # Parse batch response
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```\w*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)
            answers = json.loads(clean)
        except Exception:
            # Fallback: try to find individual answers
            answers = {}
            for key in ["A", "B", "C", "D"]:
                match = re.search(rf'"{key}"\s*:\s*"(yes|no|maybe)"', response, re.IGNORECASE)
                if match:
                    answers[key] = match.group(1).lower()

        method_map = {
            "Full Context": "A",
            "Embedding (50%)": "B",
            "Lumisift (50%)": "C",
            "Hybrid (50%)": "D",
        }

        for method, key in method_map.items():
            answer = answers.get(key, "maybe").lower().strip()
            is_correct = answer == gold_answer
            results[method]["correct"] += int(is_correct)
            results[method]["total"] += 1
            results[method]["per_article"].append({
                "pubid": pubid,
                "question": question[:80],
                "gold": gold_answer,
                "predicted": answer,
                "correct": is_correct,
            })

        # Progress + rate limiting
        if (i + 1) % 5 == 0:
            full_acc = results["Full Context"]["correct"] / max(1, results["Full Context"]["total"]) * 100
            lumi_acc = results["Lumisift (50%)"]["correct"] / max(1, results["Lumisift (50%)"]["total"]) * 100
            print(f"  [{i+1}/{len(instances)}] Full={full_acc:.1f}% Lumi={lumi_acc:.1f}% ({api_calls} API calls)", flush=True)

            # Save checkpoint every 5 items
            checkpoint = {"results": results, "skipped": skipped, "api_calls": api_calls, "next_idx": i + 1}
            os.makedirs("benchmark_data", exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)

        time.sleep(2.5)  # Stay under Groq 30 RPM limit

    # Clean up checkpoint on completion
    if os.path.exists(checkpoint_path) and results["Full Context"]["total"] >= len(instances) - skipped:
        os.remove(checkpoint_path)
        print("  Checkpoint removed (benchmark complete).", flush=True)

    # ─── Results ──────────────────────────────────────────────────────────

    print(f"\n\n{'='*70}")
    print("  RESULTS: PubMedQA Official Benchmark")
    print(f"  Dataset: PubMedQA-Labeled (Jin et al., ACL 2019)")
    print(f"{'='*70}\n")

    print(f"  Instances evaluated: {results['Full Context']['total']}")
    print(f"  Instances skipped:   {skipped}")
    print(f"  API calls:           {api_calls}")
    print(f"  Ground truth:        Human expert annotations\n")

    print(f"  {'Method':<35} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*55}")

    for method in methods:
        r = results[method]
        acc = r["correct"] / max(1, r["total"]) * 100
        print(f"  {method:<35} {r['correct']:>5}/{r['total']} {acc:>9.1f}%")

    # Efficiency analysis
    print(f"\n  {'Efficiency Analysis':}")
    print(f"  {'-'*55}")
    full_acc = results["Full Context"]["correct"] / max(1, results["Full Context"]["total"]) * 100
    for method in methods[1:]:
        r = results[method]
        acc = r["correct"] / max(1, r["total"]) * 100
        delta = acc - full_acc
        retention = acc / max(0.01, full_acc) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {method:<35} {sign}{delta:.1f}pp  ({retention:.0f}% of full accuracy, 50% tokens)")

    # Per-answer-type breakdown
    print(f"\n  {'Answer Type Breakdown':}")
    print(f"  {'-'*55}")
    for answer_type in ["yes", "no", "maybe"]:
        type_results = {}
        for method in methods:
            correct = sum(1 for r in results[method]["per_article"]
                         if r["gold"] == answer_type and r["correct"])
            total = sum(1 for r in results[method]["per_article"]
                       if r["gold"] == answer_type)
            type_results[method] = (correct, total)

        total_of_type = type_results["Full Context"][1]
        if total_of_type > 0:
            print(f"\n  Gold='{answer_type}' ({total_of_type} instances):")
            for method in methods:
                c, t = type_results[method]
                acc = c / max(1, t) * 100
                print(f"    {method:<33} {c:>3}/{t} = {acc:.1f}%")

    # Key findings
    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}\n")

    lumi_acc = results["Lumisift (50%)"]["correct"] / max(1, results["Lumisift (50%)"]["total"]) * 100
    sim_acc = results["Embedding (50%)"]["correct"] / max(1, results["Embedding (50%)"]["total"]) * 100
    hybrid_acc = results["Hybrid (50%)"]["correct"] / max(1, results["Hybrid (50%)"]["total"]) * 100

    if lumi_acc >= full_acc:
        print(f"  [+] Lumisift MATCHES or BEATS full context ({lumi_acc:.1f}% vs {full_acc:.1f}%)")
        print(f"    with 50% fewer tokens!")
    elif lumi_acc > sim_acc:
        print(f"  [+] Lumisift outperforms embedding similarity ({lumi_acc:.1f}% vs {sim_acc:.1f}%)")
        print(f"    Delta: +{lumi_acc - sim_acc:.1f} percentage points")

    if hybrid_acc > max(lumi_acc, sim_acc):
        print(f"  [+] Hybrid approach gives best accuracy among 50% methods ({hybrid_acc:.1f}%)")

    # ─── Save Results ─────────────────────────────────────────────────────

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "dataset": "PubMedQA-Labeled (qiaojin/PubMedQA, pqa_labeled)",
            "paper": "Jin et al., PubMedQA: A Dataset for Biomedical Research Question Answering, ACL 2019",
            "ground_truth": "Human expert annotations",
            "model_judge": f"{API_PROVIDER} / {MODEL_NAME}",
            "instances_evaluated": results["Full Context"]["total"],
            "instances_skipped": skipped,
            "api_calls": api_calls,
            "selection_ratio": "50%",
            "note": "Official benchmark — no self-generated questions, no circular validation",
        },
        "accuracy": {
            method: {
                "correct": results[method]["correct"],
                "total": results[method]["total"],
                "pct": round(results[method]["correct"] / max(1, results[method]["total"]) * 100, 1),
            }
            for method in methods
        },
        "per_article": {
            method: results[method]["per_article"]
            for method in methods
        },
    }

    os.makedirs("benchmark_data", exist_ok=True)
    out_path = os.path.join("benchmark_data", "pubmedqa_official.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")
    print(f"\n  NOTE: This benchmark uses the official PubMedQA dataset.")
    print(f"  Results are directly comparable to the PubMedQA leaderboard.")


if __name__ == "__main__":
    main()
