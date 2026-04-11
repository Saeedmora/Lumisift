"""
Downstream Quality Evaluation — Logical Rooms (Batch Mode)
===========================================================
Uses 3 total Gemini API calls to evaluate downstream quality.
Designed for free-tier rate limits.

Approach:
  Call 1: Generate 10 questions for 10 articles (1 API call)
  Call 2: Answer all questions with full text AND selected text (1 API call)  
  Call 3: Grade all answer pairs (1 API call)
"""

import os
import sys
import json
import time
import re
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from google import genai

from core.atom import Atom, _count_tokens
from core.axes_evaluator import SevenAxesEvaluator

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-3-flash-preview"

client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            wait = min(60, (attempt + 1) * 10)
            print(f"  API error (attempt {attempt+1}): {e}")
            print(f"  Waiting {wait}s...")
            time.sleep(wait)
    return ""


# ─── Load Articles ──────────────────────────────────────────────────────────

print("=" * 70)
print("  DOWNSTREAM QUALITY EVALUATION — BATCH MODE")
print("  Logical Rooms — Axes-Driven Selection vs Full Text")
print("=" * 70)
print()

articles_path = os.path.join("benchmark_data", "pubmed_articles.json")
with open(articles_path, "r", encoding="utf-8") as f:
    all_articles = json.load(f)

# 10 articles with >100 words
articles = [a for a in all_articles if len(a.get("abstract", "").split()) > 100][:10]
print(f"Loaded {len(articles)} articles\n")

# ─── Initialize Evaluator ──────────────────────────────────────────────────

print("Initializing heuristic evaluator...")
evaluator = SevenAxesEvaluator(use_llm=False)
print("Ready.\n")

# ─── Process All Articles ──────────────────────────────────────────────────

print("Step 1: Processing articles through 7-axis evaluator...")

eval_data = []

for i, article in enumerate(articles):
    abstract = article.get("abstract", "")
    title = article.get("title", "")[:60]
    pmid = article.get("pmid", "?")

    # Split into sentence-based chunks
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract)
    chunks = []
    current = ""
    for s in sentences:
        current += (" " if current else "") + s
        if len(current.split()) >= 25:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())

    if len(chunks) < 2:
        continue

    # Evaluate each chunk
    scored_chunks = []
    for chunk in chunks:
        axes, category = evaluator.evaluate(chunk)
        rel = abs(axes.get("relevance", 0))
        risk = abs(axes.get("risk", 0))
        trust = axes.get("trust", 0.5)
        specificity = axes.get("specificity", 0.0)

        # Same formula as pipeline.py select_context:
        # score = rel * (1 + risk) * (0.5 + trust * 0.5) * t_boost * s_boost
        s_boost = 1.0 + specificity * 0.8  # 1.0x to 1.8x for quantitative data
        tension = rel * (1 + risk) * (1 - trust * 0.5) * s_boost
        scored_chunks.append({"text": chunk, "tension": tension, "axes": axes, "specificity": specificity})

    # Select top 50% by tension (with specificity boost)
    scored_chunks.sort(key=lambda x: x["tension"], reverse=True)
    n_select = max(1, len(scored_chunks) // 2)
    selected = scored_chunks[:n_select]

    full_text = abstract
    selected_text = " ".join(s["text"] for s in selected)

    full_tokens = _count_tokens(full_text)
    sel_tokens = _count_tokens(selected_text)
    reduction = 1.0 - (sel_tokens / max(1, full_tokens))

    eval_data.append({
        "idx": i,
        "pmid": pmid,
        "title": title,
        "full_text": full_text,
        "selected_text": selected_text,
        "full_tokens": full_tokens,
        "sel_tokens": sel_tokens,
        "reduction": reduction,
        "n_chunks": len(chunks),
        "n_selected": n_select,
    })
    print(f"  [{i+1}] {title}... -> {len(chunks)} chunks, top {n_select} selected ({reduction*100:.0f}% reduction)")

print(f"\n{len(eval_data)} articles processed.\n")

# ─── Mega Call 1: Generate Questions + Answer with FULL text ────────────────

print("Step 2: Generating questions and answering with FULL text (1 API call)...")

articles_block = ""
for d in eval_data:
    articles_block += f"\n---ARTICLE {d['idx']}---\nPMID: {d['pmid']}\nTitle: {d['title']}\n\nFull Text:\n{d['full_text']}\n"

prompt_qa_full = f"""You are a protein engineering expert. For each of the following articles:
1. Generate ONE specific scientific question answerable from the text
2. Answer that question using ONLY the full text provided

Return a JSON array with exactly {len(eval_data)} objects, one per article.
Each object must have: "idx", "question", "answer_full"

Articles:
{articles_block}

Return ONLY the JSON array, no markdown formatting, no code fences."""

time.sleep(5)  # Wait before first call
response_full = ask_gemini(prompt_qa_full)

if not response_full:
    print("ERROR: Failed to generate questions. API quota may be exhausted.")
    print("Try again in a few minutes.")
    sys.exit(1)

# Parse response
try:
    # Clean response
    clean = response_full.strip()
    if clean.startswith("```"):
        clean = re.sub(r'^```\w*\n?', '', clean)
        clean = re.sub(r'\n?```$', '', clean)
    qa_full = json.loads(clean)
    print(f"  Got {len(qa_full)} question+answer pairs.\n")
except Exception as e:
    print(f"Parse error: {e}")
    print(f"Raw response:\n{response_full[:500]}")
    # Try to save what we can
    qa_full = []

# ─── Mega Call 2: Answer with SELECTED text ─────────────────────────────────

if qa_full:
    print("Step 3: Answering same questions with SELECTED text (1 API call)...")

    selected_block = ""
    for d in eval_data:
        q = next((qa["question"] for qa in qa_full if qa.get("idx") == d["idx"]), "")
        selected_block += f"\n---ARTICLE {d['idx']}---\nQuestion: {q}\n\nSelected Context (tension-ranked top passages):\n{d['selected_text']}\n"

    prompt_qa_sel = f"""You are a protein engineering expert. For each article below,
answer the given question using ONLY the selected context provided.
Be specific and cite details from the text.

Return a JSON array with exactly {len(eval_data)} objects.
Each object must have: "idx", "answer_selected"

Articles:
{selected_block}

Return ONLY the JSON array, no markdown formatting, no code fences."""

    time.sleep(10)  # Longer pause between calls
    response_sel = ask_gemini(prompt_qa_sel)

    try:
        clean_sel = response_sel.strip()
        if clean_sel.startswith("```"):
            clean_sel = re.sub(r'^```\w*\n?', '', clean_sel)
            clean_sel = re.sub(r'\n?```$', '', clean_sel)
        qa_sel = json.loads(clean_sel)
        print(f"  Got {len(qa_sel)} selected-text answers.\n")
    except Exception as e:
        print(f"Parse error: {e}")
        qa_sel = []

# ─── Mega Call 3: Grade all pairs ───────────────────────────────────────────

if qa_full and qa_sel:
    print("Step 4: Grading all answer pairs (1 API call)...")

    grade_block = ""
    for d in eval_data:
        qf = next((qa for qa in qa_full if qa.get("idx") == d["idx"]), {})
        qs = next((qa for qa in qa_sel if qa.get("idx") == d["idx"]), {})
        if qf and qs:
            grade_block += f"""
---ARTICLE {d['idx']} (PMID: {d['pmid']})---
Ground Truth Abstract: {d['full_text'][:300]}...
Question: {qf.get('question', '')}

Answer A (full text): {qf.get('answer_full', '')}
Answer B (selected text, {d['reduction']*100:.0f}% reduction): {qs.get('answer_selected', '')}
"""

    prompt_grade = f"""You are an expert scientific evaluator. Grade each pair of answers below.
For each article, grade BOTH answers on 4 dimensions (1-5 scale):
- accuracy: factual correctness compared to the abstract
- completeness: coverage of key relevant points
- relevance: focus on the question without irrelevant info
- conciseness: appropriate brevity without unnecessary padding

Return a JSON array with one object per article.
Each object must have: "idx", "grade_full" (dict with accuracy/completeness/relevance/conciseness), "grade_selected" (same dict)

{grade_block}

Return ONLY the JSON array, no markdown formatting, no code fences."""

    time.sleep(10)
    response_grade = ask_gemini(prompt_grade)

    try:
        clean_grade = response_grade.strip()
        if clean_grade.startswith("```"):
            clean_grade = re.sub(r'^```\w*\n?', '', clean_grade)
            clean_grade = re.sub(r'\n?```$', '', clean_grade)
        grades = json.loads(clean_grade)
        print(f"  Got {len(grades)} grade pairs.\n")
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw: {response_grade[:500]}")
        grades = []

# ─── Compute Results ────────────────────────────────────────────────────────

print("=" * 70)
print("  RESULTS")
print("=" * 70)
print()

if not grades:
    print("No grades collected. API quota exhausted.")
    sys.exit(1)

dimensions = ["accuracy", "completeness", "relevance", "conciseness"]
full_scores = {d: [] for d in dimensions}
sel_scores = {d: [] for d in dimensions}

results = []
for g in grades:
    idx = g.get("idx", -1)
    gf = g.get("grade_full", {})
    gs = g.get("grade_selected", {})
    d = next((x for x in eval_data if x["idx"] == idx), None)
    qf = next((qa for qa in qa_full if qa.get("idx") == idx), {})

    if not d:
        continue

    for dim in dimensions:
        fv = gf.get(dim, 0)
        sv = gs.get(dim, 0)
        if fv > 0 and sv > 0:
            full_scores[dim].append(fv)
            sel_scores[dim].append(sv)

    results.append({
        "pmid": d["pmid"],
        "title": d["title"],
        "question": qf.get("question", ""),
        "full_tokens": d["full_tokens"],
        "selected_tokens": d["sel_tokens"],
        "reduction_pct": round(d["reduction"] * 100, 1),
        "grade_full": gf,
        "grade_selected": gs,
    })

    print(f"  PMID:{d['pmid']} - {d['title']}")
    print(f"    FULL:     A={gf.get('accuracy','?')} C={gf.get('completeness','?')} R={gf.get('relevance','?')} Cn={gf.get('conciseness','?')}  [{d['full_tokens']} tok]")
    print(f"    SELECTED: A={gs.get('accuracy','?')} C={gs.get('completeness','?')} R={gs.get('relevance','?')} Cn={gs.get('conciseness','?')}  [{d['sel_tokens']} tok]")
    print()

# Summary statistics
n = len(results)
total_full = sum(r["full_tokens"] for r in results)
total_sel = sum(r["selected_tokens"] for r in results)
avg_reduction = np.mean([r["reduction_pct"] for r in results]) if results else 0

print(f"Articles evaluated:     {n}")
print(f"Total full tokens:      {total_full}")
print(f"Total selected tokens:  {total_sel}")
print(f"Avg context reduction:  {avg_reduction:.1f}%")
print(f"Token savings:          {total_full - total_sel} tokens ({(1-total_sel/max(1,total_full))*100:.1f}%)")
print()

print(f"{'Dimension':<16} {'Full Text':>10} {'Selected':>10} {'Delta':>8} {'Verdict':>12}")
print("-" * 60)

summary = {}
for d in dimensions:
    if full_scores[d]:
        fm = np.mean(full_scores[d])
        sm = np.mean(sel_scores[d])
        delta = sm - fm
        if delta > 0.15:
            verdict = "SELECTED+"
        elif delta < -0.15:
            verdict = "FULL+"
        else:
            verdict = "TIE"
        print(f"{d:<16} {fm:>10.2f} {sm:>10.2f} {delta:>+8.2f} {verdict:>12}")
        summary[d] = {"full": round(fm, 3), "selected": round(sm, 3), "delta": round(delta, 3), "verdict": verdict.strip()}

# Composite
full_composite = np.mean([np.mean(full_scores[d]) for d in dimensions if full_scores[d]])
sel_composite = np.mean([np.mean(sel_scores[d]) for d in dimensions if sel_scores[d]])
composite_delta = sel_composite - full_composite

if composite_delta > 0.15:
    overall = "SELECTED WINS"
elif composite_delta < -0.15:
    overall = "FULL WINS"
else:
    overall = "STATISTICALLY EQUIVALENT"

print("-" * 60)
print(f"{'COMPOSITE':<16} {full_composite:>10.2f} {sel_composite:>10.2f} {composite_delta:>+8.2f} {overall:>12}")
print()

# Efficiency
quality_ratio_full = full_composite / max(1, total_full) * 1000
quality_ratio_sel = sel_composite / max(1, total_sel) * 1000
efficiency_gain = (quality_ratio_sel / max(0.0001, quality_ratio_full) - 1) * 100

print("--- EFFICIENCY ANALYSIS ---")
print(f"Quality/1000 tokens (full):     {quality_ratio_full:.4f}")
print(f"Quality/1000 tokens (selected): {quality_ratio_sel:.4f}")
print(f"Efficiency gain:                {efficiency_gain:+.1f}%")
print()

# ─── Save Results ───────────────────────────────────────────────────────────

output = {
    "metadata": {
        "date": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "methodology": "Batch QA evaluation: generate questions -> answer with full/selected text -> grade by Gemini judge",
        "articles_evaluated": n,
        "api_calls": 3,
    },
    "token_stats": {
        "total_full_tokens": total_full,
        "total_selected_tokens": total_sel,
        "avg_reduction_pct": round(avg_reduction, 1),
    },
    "quality_scores": summary,
    "composite": {
        "full_text": round(full_composite, 3),
        "selected_text": round(sel_composite, 3),
        "delta": round(composite_delta, 3),
        "verdict": overall,
    },
    "efficiency": {
        "quality_per_1k_tokens_full": round(quality_ratio_full, 4),
        "quality_per_1k_tokens_selected": round(quality_ratio_sel, 4),
        "efficiency_gain_pct": round(efficiency_gain, 1),
    },
    "per_article": results,
}

output_path = os.path.join("benchmark_data", "downstream_quality.json")
os.makedirs("benchmark_data", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")
print()
print("=" * 70)
print("  CONCLUSION")
print("=" * 70)
print()
print(f"  Axes-driven selection reduced context by {avg_reduction:.1f}% on average.")
if overall == "STATISTICALLY EQUIVALENT":
    print(f"  Answer quality is STATISTICALLY EQUIVALENT (delta={composite_delta:+.3f}).")
    print(f"  -> Same quality AI answers using {avg_reduction:.0f}% fewer tokens.")
    print(f"  -> Efficiency gain: {efficiency_gain:+.1f}% more quality per token.")
elif "SELECTED" in overall:
    print(f"  Selected text produced HIGHER quality answers (delta={composite_delta:+.3f}).")
    print(f"  -> Better answers AND {avg_reduction:.0f}% fewer tokens.")
elif "FULL" in overall:
    print(f"  Full text produced higher quality answers (delta={composite_delta:+.3f}).")
    print(f"  -> Trade-off: quality vs {avg_reduction:.0f}% cost savings.")
print()
