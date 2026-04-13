"""
SciFact Benchmark — Scientific Claim Verification via BEIR
============================================================
Uses the official SciFact dataset via BEIR format (Wadden et al., EMNLP 2020)
to test whether Lumisift's context selection preserves enough evidence for
scientific claim verification.

Dataset structure (BeIR format):
  - Queries: 1,109 scientific claims
  - Corpus: 5,183 scientific abstracts  
  - Qrels: 339 relevance judgments (claim → document)

Methodology:
  1. Load SciFact claims + corpus + relevance judgments
  2. For each claim with a relevant document:
     a. Split the abstract into sentences
     b. Select 50% of sentences with each method
     c. Verify the claim using LLM: SUPPORTS / REFUTES / NOT_ENOUGH_INFO
     d. Compare all methods against the same ground truth
  3. Additionally: measure evidence sentence retention rate
     (does the selected 50% still contain the relevant content?)

This tests a fundamentally different capability than PubMedQA:
  - PubMedQA: "Can you answer questions with less context?"
  - SciFact:  "Can you preserve EVIDENCE for scientific reasoning?"

Paper: Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP 2020
Dataset: https://huggingface.co/datasets/BeIR/scifact
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

# ─── API Configuration (Groq > xAI/Grok > Gemini) ────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GROQ_API_KEY:
    from groq import Groq
    api_client = Groq(api_key=GROQ_API_KEY)
    MODEL_NAME = "llama-3.1-8b-instant"
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

# Limit for faster testing (set to None for full benchmark)
MAX_CLAIMS = None  # e.g. 30 for quick test


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
            if "429" in err_str and "FreeTier" in err_str:
                print(f"  QUOTA EXHAUSTED. Saving checkpoint.", flush=True)
                return "__QUOTA_EXHAUSTED__"
            wait = min(60, (attempt + 1) * 10)
            # For rate limit 429, wait the suggested time
            if "429" in err_str:
                delay_match = re.search(r'retry.after.*?(\d+)', err_str, re.IGNORECASE)
                if delay_match:
                    wait = int(delay_match.group(1)) + 2
                else:
                    wait = max(wait, 30)  # conservative wait for rate limits
            print(f"  API error (attempt {attempt+1}): {str(e)[:200]}", flush=True)
            print(f"  Waiting {wait}s...", flush=True)
            time.sleep(wait)
    return ""


# ─── Selection Methods ────────────────────────────────────────────────────

def select_by_similarity(query, chunks, embedder, top_k):
    """Standard RAG: cosine similarity selection."""
    if len(chunks) <= top_k:
        return list(range(len(chunks)))
    query_emb = embedder.embed(query)
    chunk_embs = embedder.embed_many(chunks)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    c = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    sims = c @ q
    return np.argsort(sims)[::-1][:top_k].tolist()


def select_by_lumisift(chunks, evaluator, top_k):
    """Lumisift: multi-axis with specificity boost."""
    if len(chunks) <= top_k:
        return list(range(len(chunks)))
    scored = []
    for i, chunk in enumerate(chunks):
        axes, _ = evaluator.evaluate(chunk)
        rel = abs(axes.get("relevance", 0))
        risk = abs(axes.get("risk", 0))
        trust = axes.get("trust", 0.5)
        spec = axes.get("specificity", 0.0)
        s_boost = 1.0 + spec * 0.8
        score = rel * (1 + risk) * (0.5 + trust * 0.5) * s_boost
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:top_k]]


def select_hybrid(query, chunks, embedder, evaluator, top_k, alpha=0.3):
    """Hybrid: alpha * similarity + (1-alpha) * lumisift."""
    if len(chunks) <= top_k:
        return list(range(len(chunks)))

    query_emb = embedder.embed(query)
    chunk_embs = embedder.embed_many(chunks)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    c = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    sim_scores = c @ q

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
    print("  SCIFACT CLAIM VERIFICATION BENCHMARK")
    print("  Expert-Annotated Evidence (Wadden et al., EMNLP 2020)")
    print("  Can Lumisift preserve evidence for scientific reasoning?")
    print("=" * 70)
    print()

    if not GROQ_API_KEY and not XAI_API_KEY and not GEMINI_API_KEY:
        print("ERROR: No API key found. Set GROQ_API_KEY, XAI_API_KEY, or GEMINI_API_KEY in .env")
        sys.exit(1)

    # ─── Step 1: Load SciFact Dataset via BEIR format ─────────────────────

    print("Step 1: Loading SciFact dataset from HuggingFace (BEIR format)...")

    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    # Build lookup dicts
    queries = {item["_id"]: item["text"] for item in queries_ds}
    corpus = {item["_id"]: {"title": item["title"], "text": item["text"]} for item in corpus_ds}

    # Build claim → relevant doc mapping
    claim_to_docs = {}
    for qrel in qrels_ds:
        qid = str(qrel["query-id"])
        cid = str(qrel["corpus-id"])
        score = qrel["score"]
        if score > 0:  # Only positive relevance
            if qid not in claim_to_docs:
                claim_to_docs[qid] = []
            claim_to_docs[qid].append(cid)

    print(f"  Queries (claims): {len(queries)}")
    print(f"  Corpus (abstracts): {len(corpus)}")
    print(f"  Relevance judgments: {len(qrels_ds)}")
    print(f"  Claims with relevant docs: {len(claim_to_docs)}")
    print()

    # Initialize models
    print("Loading models...")
    embedder = EmbeddingService()
    evaluator = SevenAxesEvaluator(use_llm=False)
    print("Ready.\n")

    # ─── Step 2: Evaluate each claim ──────────────────────────────────────

    print("Step 2: Evaluating claims...\n")

    methods = ["Full Context", "Embedding (50%)", "Lumisift (50%)", "Hybrid (50%)"]
    results = {m: {"correct": 0, "total": 0, "per_claim": []} for m in methods}
    evidence_retention = {m: [] for m in methods}

    evaluated = 0
    skipped = 0

    # Process each claim that has a relevant document
    claim_ids = list(claim_to_docs.keys())
    if MAX_CLAIMS is not None:
        claim_ids = claim_ids[:MAX_CLAIMS]

    for ci, qid in enumerate(claim_ids):
        claim_text = queries.get(qid, "")
        if not claim_text:
            skipped += 1
            continue

        relevant_doc_ids = claim_to_docs[qid]

        # Get the primary relevant document
        doc_id = relevant_doc_ids[0]
        doc = corpus.get(doc_id, None)
        if not doc:
            skipped += 1
            continue

        abstract_text = doc["text"]
        doc_title = doc["title"]

        # Split abstract into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) < 3:
            skipped += 1
            continue

        evaluated += 1
        n_select = max(1, len(sentences) // 2)
        full_text = " ".join(sentences)

        # ── Select with each method ───────────────────────────────────

        sim_idx = select_by_similarity(claim_text, sentences, embedder, n_select)
        lumi_idx = select_by_lumisift(sentences, evaluator, n_select)
        hybrid_idx = select_hybrid(claim_text, sentences, embedder, evaluator, n_select)

        sim_text = " ".join(sentences[j] for j in sorted(sim_idx))
        lumi_text = " ".join(sentences[j] for j in sorted(lumi_idx))
        hybrid_text = " ".join(sentences[j] for j in sorted(hybrid_idx))

        # ── Verify claim with LLM ─────────────────────────────────────

        # Ground truth: the document is RELEVANT (score > 0 in qrels)
        # We'll test: can each method still support the claim verification?
        # The "correct" verification is that the text is relevant to the claim.

        batch_prompt = f"""You are a scientific fact-checker. For the scientific claim below,
determine whether each context SUPPORTS the claim, REFUTES it, or provides 
NOT_ENOUGH_INFO to make a determination.

Scientific Claim: {claim_text}

---CONTEXT A (full abstract)---
Title: {doc_title}
{full_text[:1500]}

---CONTEXT B (embedding-selected, 50% of sentences)---
{sim_text[:1000]}

---CONTEXT C (lumisift-selected, 50% of sentences)---
{lumi_text[:1000]}

---CONTEXT D (hybrid-selected, 50% of sentences)---
{hybrid_text[:1000]}

Return a JSON object with keys "A", "B", "C", "D".
Each value must be exactly one of: "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"
Return ONLY the JSON, no markdown, no code fences."""

        response = ask_llm(batch_prompt)

        # Check for quota exhaustion
        if response == "__QUOTA_EXHAUSTED__":
            checkpoint = {"results": {m: results[m] for m in methods},
                          "evidence_retention": {m: evidence_retention[m] for m in methods},
                          "evaluated": evaluated, "skipped": skipped, "next_ci": ci}
            os.makedirs("benchmark_data", exist_ok=True)
            with open(os.path.join("benchmark_data", "scifact_checkpoint.json"), "w") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  Checkpoint saved at claim {ci}. Re-run to continue.", flush=True)
            break

        # Parse response
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```\w*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)
            answers = json.loads(clean)
        except Exception:
            answers = {}
            for key in ["A", "B", "C", "D"]:
                match = re.search(rf'"{key}"\s*:\s*"(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)"',
                                  response, re.IGNORECASE)
                if match:
                    answers[key] = match.group(1).upper()

        # The gold standard: full context answer is our reference
        # (Since qrels confirms the doc is relevant, the full-context answer
        # is our "ground truth" — and we measure how selection changes it)
        full_answer = answers.get("A", "NOT_ENOUGH_INFO").upper()
        if "SUPPORT" in full_answer:
            full_answer = "SUPPORTS"
        elif "REFUTE" in full_answer:
            full_answer = "REFUTES"
        else:
            full_answer = "NOT_ENOUGH_INFO"

        # Record full context as always correct (it's the reference)
        results["Full Context"]["correct"] += 1
        results["Full Context"]["total"] += 1
        evidence_retention["Full Context"].append(1.0)

        method_map = {
            "Embedding (50%)": ("B", sim_idx),
            "Lumisift (50%)": ("C", lumi_idx),
            "Hybrid (50%)": ("D", hybrid_idx),
        }

        for method, (key, idx) in method_map.items():
            pred = answers.get(key, "NOT_ENOUGH_INFO").upper()
            if "SUPPORT" in pred:
                pred = "SUPPORTS"
            elif "REFUTE" in pred:
                pred = "REFUTES"
            else:
                pred = "NOT_ENOUGH_INFO"

            is_correct = (pred == full_answer)
            results[method]["correct"] += int(is_correct)
            results[method]["total"] += 1

            # Compute content retention (how much of the important text is kept)
            selected_text = " ".join(sentences[j] for j in idx)
            words_full = set(full_text.lower().split())
            words_sel = set(selected_text.lower().split())
            content_overlap = len(words_full & words_sel) / max(1, len(words_full))
            evidence_retention[method].append(content_overlap)

            results[method]["per_claim"].append({
                "claim_id": qid,
                "claim": claim_text[:80],
                "full_verdict": full_answer,
                "method_verdict": pred,
                "agreement": is_correct,
            })

        results["Full Context"]["per_claim"].append({
            "claim_id": qid,
            "claim": claim_text[:80],
            "full_verdict": full_answer,
            "method_verdict": full_answer,
            "agreement": True,
        })

        # Progress
        if (ci + 1) % 10 == 0:
            lumi_cor = results["Lumisift (50%)"]["correct"]
            lumi_tot = max(1, results["Lumisift (50%)"]["total"])
            print(f"  [{ci+1}/{len(claim_ids)}] Evaluated {evaluated}, "
                  f"Lumisift agreement: {lumi_cor/lumi_tot*100:.1f}%")

        time.sleep(2.5)  # Stay under Groq 30 RPM limit

    # ─── Results ──────────────────────────────────────────────────────────

    print(f"\n\n{'='*70}")
    print("  RESULTS: SciFact Claim Verification Benchmark")
    print(f"  Dataset: SciFact (Wadden et al., EMNLP 2020)")
    print(f"{'='*70}\n")

    print(f"  Claims evaluated: {evaluated}")
    print(f"  Claims skipped:   {skipped}")
    print(f"  Reference:        Full-context LLM verdict\n")

    print(f"  {'Method':<35} {'Agreement':>12} {'Rate':>8}")
    print(f"  {'-'*58}")

    for method in methods:
        r = results[method]
        rate = r["correct"] / max(1, r["total"]) * 100
        print(f"  {method:<35} {r['correct']:>5}/{r['total']:>5} {rate:>7.1f}%")

    # Evidence / content retention
    print(f"\n  Content Retention (word overlap with full text):")
    print(f"  {'-'*58}")
    for method in methods:
        er = evidence_retention[method]
        if er:
            mean_ret = np.mean(er) * 100
            print(f"  {method:<35} {mean_ret:>9.1f}%")

    # Verdict distribution
    print(f"\n  Verdict Distribution (Full Context):")
    print(f"  {'-'*58}")
    verdicts = [r["full_verdict"] for r in results["Full Context"]["per_claim"]]
    for v in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]:
        count = verdicts.count(v)
        print(f"  {v:<25} {count:>5} ({count/max(1,len(verdicts))*100:.1f}%)")

    # Per-verdict agreement
    print(f"\n  Agreement by Full-Context Verdict:")
    print(f"  {'-'*58}")
    for verdict_type in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]:
        relevant_indices = [i for i, r in enumerate(results["Full Context"]["per_claim"])
                           if r["full_verdict"] == verdict_type]
        if not relevant_indices:
            continue
        print(f"\n  When full context says '{verdict_type}' ({len(relevant_indices)} claims):")
        for method in methods[1:]:  # Skip full context
            method_claims = results[method]["per_claim"]
            agreed = sum(1 for i in relevant_indices
                        if i < len(method_claims) and method_claims[i]["agreement"])
            rate = agreed / max(1, len(relevant_indices)) * 100
            print(f"    {method:<33} {agreed:>3}/{len(relevant_indices)} = {rate:.1f}%")

    # Key findings
    print(f"\n{'='*70}")
    print("  KEY FINDINGS")
    print(f"{'='*70}\n")

    for method in methods[1:]:
        r = results[method]
        rate = r["correct"] / max(1, r["total"]) * 100
        er = np.mean(evidence_retention[method]) * 100 if evidence_retention[method] else 0
        print(f"  {method:<35}")
        print(f"    Verdict agreement with full text: {rate:.1f}%")
        print(f"    Content retention:                {er:.1f}%")
        print()

    # ─── Save Results ─────────────────────────────────────────────────────

    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "dataset": "SciFact via BEIR (BeIR/scifact)",
            "paper": "Wadden et al., Fact or Fiction: Verifying Scientific Claims, EMNLP 2020",
            "ground_truth": "Full-context LLM verdict (document confirmed relevant by human qrels)",
            "model_judge": f"{API_PROVIDER} / {MODEL_NAME}",
            "claims_evaluated": evaluated,
            "claims_skipped": skipped,
            "selection_ratio": "50%",
            "note": "Agreement measures how often 50%-selection produces the same "
                    "claim verdict as full context — on documents confirmed relevant by experts.",
        },
        "agreement_rates": {
            method: {
                "agreed": results[method]["correct"],
                "total": results[method]["total"],
                "pct": round(results[method]["correct"] / max(1, results[method]["total"]) * 100, 1),
            }
            for method in methods
        },
        "content_retention": {
            method: {
                "mean_pct": round(np.mean(evidence_retention[method]) * 100, 1)
                if evidence_retention[method] else None,
            }
            for method in methods
        },
        "per_claim": {
            method: results[method]["per_claim"]
            for method in methods
        },
    }

    os.makedirs("benchmark_data", exist_ok=True)
    out_path = os.path.join("benchmark_data", "scifact_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")
    print(f"\n  NOTE: This benchmark uses the official SciFact dataset (EMNLP 2020).")
    print(f"  It tests whether Lumisift can select the 50% of sentences that")
    print(f"  preserve the same scientific verdict as the full abstract.")


if __name__ == "__main__":
    main()
