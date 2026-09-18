"""
Reply Quality Evaluation — Blinded, Order-Randomized, Dual-Judge (Task 9)
- n >= 150 stratified examples
- Blinded judging: Reply A / Reply B in random order
- Two different judge models for inter-rater agreement
- Cohen's kappa + Pearson between judges
"""
import sys, io
from pathlib import Path
import json
import time
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *
from src.pipeline import llm_cache
from src.pipeline.reply_generator import ReplyGenerator
from groq import Groq


def evaluate_reply_blinded(client, model, customer_text, reply_a, reply_b, actual_reply):
    """Evaluate two replies in blinded fashion (no system/baseline labels)."""
    prompt = f"""You are evaluating customer support reply quality for Amazon.

Customer message: {customer_text}
Actual brand reply (reference): {actual_reply}

Evaluate each reply independently on these dimensions (1=terrible, 5=excellent):
1. Relevance: Does it address the customer's specific issue?
2. Grounding: Is it consistent with how a real support agent would respond?
3. Tone: Is it professional, empathetic, and brand-appropriate?
4. Actionability: Does it provide a clear next step?

Reply A: {reply_a}
Reply B: {reply_b}

Output ONLY valid JSON with this exact structure:
{{"reply_a": {{"relevance": X, "grounding": X, "tone": X, "actionability": X, "overall": X}}, "reply_b": {{"relevance": X, "grounding": X, "tone": X, "actionability": X, "overall": X}}, "preferred": "A or B or tie", "reasoning": "brief explanation"}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = llm_cache.cached_completion(
            client, model,
            messages=messages,
            temperature=TEMPERATURE_JUDGE,
            max_tokens=MAX_TOKENS_JUDGE,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)

        def extract_scores(d):
            return {
                "relevance": int(d.get("relevance", 3)),
                "grounding": int(d.get("grounding", 3)),
                "tone": int(d.get("tone", 3)),
                "actionability": int(d.get("actionability", 3)),
                "overall": int(d.get("overall", 3)),
            }

        return {
            "reply_a": extract_scores(result.get("reply_a", {})),
            "reply_b": extract_scores(result.get("reply_b", {})),
            "preferred": result.get("preferred", "tie"),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        default = {"relevance": 3, "grounding": 3, "tone": 3, "actionability": 3, "overall": 3}
        return {
            "reply_a": default.copy(),
            "reply_b": default.copy(),
            "preferred": "tie",
            "reasoning": f"Error: {e}",
        }


def bootstrap_ci_mean(scores, n_bootstrap=1000, ci=0.95):
    """Bootstrap CI for a mean score."""
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    scores = np.array(scores)
    means = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(scores), len(scores), replace=True)
        means.append(np.mean(scores[indices]))
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(scores), lower, upper


def main():
    print("=" * 70)
    print("REPLY QUALITY EVALUATION — Blinded, Dual-Judge")
    print("=" * 70)

    with open(GOLDEN_SET_LABELLED, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    # Stratified sample of 150+ examples
    from collections import Counter
    intent_counts = Counter(item['true_intent'] for item in golden_set)
    n_target = min(150, len(golden_set))

    # Proportional stratified sampling
    random.seed(42)
    by_intent = {}
    for item in golden_set:
        intent = item['true_intent']
        if intent not in by_intent:
            by_intent[intent] = []
        by_intent[intent].append(item)

    selected = []
    for intent, items in by_intent.items():
        quota = max(1, int(len(items) / len(golden_set) * n_target))
        random.shuffle(items)
        selected.extend(items[:quota])

    # Top up if needed
    remaining = [item for item in golden_set if item not in selected]
    random.shuffle(remaining)
    while len(selected) < n_target and remaining:
        selected.append(remaining.pop())

    print(f"Selected {len(selected)} examples (target: {n_target})")
    print(f"Intent distribution: {Counter(item['true_intent'] for item in selected)}")

    generator = ReplyGenerator()
    client = Groq(api_key=GROQ_API_KEY)

    results = []
    errors = 0

    for i, item in enumerate(tqdm(selected, desc="Evaluating Replies")):
        customer_text = item.get('customer_text', '')
        actual_reply = item.get('brand_reply_actual', '')
        intent = item.get('true_intent', 'general_inquiry')

        # Generate system reply (RAG-based)
        try:
            result = generator.generate_reply(customer_text, intent)
            system_reply = result['reply']
        except Exception as e:
            system_reply = "We apologize for the inconvenience. Please try again or contact us via DM."
            errors += 1
        time.sleep(1)

        # Generate baseline reply (template)
        baseline_reply = generator.generate_reply_baseline(customer_text, intent)

        # Randomize order: system as A or B
        system_is_a = random.random() < 0.5
        if system_is_a:
            reply_a, reply_b = system_reply, baseline_reply
        else:
            reply_a, reply_b = baseline_reply, system_reply

        # Judge 1
        judge1_result = evaluate_reply_blinded(
            client, GROQ_MODEL_JUDGE, customer_text, reply_a, reply_b, actual_reply
        )
        time.sleep(1.5)

        # Judge 2
        judge2_result = evaluate_reply_blinded(
            client, GROQ_MODEL_JUDGE_2, customer_text, reply_a, reply_b, actual_reply
        )
        time.sleep(1.5)

        # Map back to system/baseline scores
        if system_is_a:
            sys_judge1 = judge1_result['reply_a']
            base_judge1 = judge1_result['reply_b']
            sys_judge2 = judge2_result['reply_a']
            base_judge2 = judge2_result['reply_b']
        else:
            sys_judge1 = judge1_result['reply_b']
            base_judge1 = judge1_result['reply_a']
            sys_judge2 = judge2_result['reply_b']
            base_judge2 = judge2_result['reply_a']

        results.append({
            "idx": i,
            "customer_text": customer_text,
            "actual_reply": actual_reply,
            "system_reply": system_reply,
            "baseline_reply": baseline_reply,
            "system_is_a": system_is_a,
            "system_eval_1": sys_judge1,
            "system_eval_2": sys_judge2,
            "baseline_eval_1": base_judge1,
            "baseline_eval_2": base_judge2,
            "judge1_preferred": judge1_result['preferred'],
            "judge2_preferred": judge2_result['preferred'],
            "judge1_reasoning": judge1_result['reasoning'],
            "judge2_reasoning": judge2_result['reasoning'],
        })

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{len(selected)}, errors: {errors}", flush=True)

    # ── Analysis ──────────────────────────────────────────────────
    dims = ["relevance", "grounding", "tone", "actionability", "overall"]

    print(f"\n{'=' * 70}")
    print(f"RESULTS (n={len(results)}, errors={errors})")
    print(f"{'=' * 70}")

    # Average scores with CIs
    print("\n--- Average Scores (Judge 1) ---")
    print(f"{'Dimension':<15} {'System':>10} {'95% CI':>20} {'Baseline':>10} {'95% CI':>20}")
    for d in dims:
        sys_scores = [r['system_eval_1'][d] for r in results]
        base_scores = [r['baseline_eval_1'][d] for r in results]
        sys_mean, sys_lo, sys_hi = bootstrap_ci_mean(sys_scores)
        base_mean, base_lo, base_hi = bootstrap_ci_mean(base_scores)
        print(f"{d:<15} {sys_mean:>10.2f} ({sys_lo:.2f}-{sys_hi:.2f}){' ':>5} {base_mean:>10.2f} ({base_lo:.2f}-{base_hi:.2f})")

    print("\n--- Average Scores (Judge 2) ---")
    print(f"{'Dimension':<15} {'System':>10} {'95% CI':>20} {'Baseline':>10} {'95% CI':>20}")
    for d in dims:
        sys_scores = [r['system_eval_2'][d] for r in results]
        base_scores = [r['baseline_eval_2'][d] for r in results]
        sys_mean, sys_lo, sys_hi = bootstrap_ci_mean(sys_scores)
        base_mean, base_lo, base_hi = bootstrap_ci_mean(base_scores)
        print(f"{d:<15} {sys_mean:>10.2f} ({sys_lo:.2f}-{sys_hi:.2f}){' ':>5} {base_mean:>10.2f} ({base_lo:.2f}-{base_hi:.2f})")

    # Combined average (both judges)
    print("\n--- Combined Average (Both Judges) ---")
    for d in dims:
        sys_scores = [(r['system_eval_1'][d] + r['system_eval_2'][d]) / 2 for r in results]
        base_scores = [(r['baseline_eval_1'][d] + r['baseline_eval_2'][d]) / 2 for r in results]
        sys_mean, sys_lo, sys_hi = bootstrap_ci_mean(sys_scores)
        base_mean, base_lo, base_hi = bootstrap_ci_mean(base_scores)
        delta = sys_mean - base_mean
        winner = "SYS" if delta > 0 else "BASE" if delta < 0 else "TIE"
        print(f"{d:<15} Sys={sys_mean:.2f} ({sys_lo:.2f}-{sys_hi:.2f})  Base={base_mean:.2f} ({base_lo:.2f}-{base_hi:.2f})  Δ={delta:+.2f} [{winner}]")

    # Inter-judge agreement
    print(f"\n--- Inter-Judge Agreement ---")
    for d in dims:
        j1 = [r['system_eval_1'][d] for r in results]
        j2 = [r['system_eval_2'][d] for r in results]

        if len(set(j1)) > 1 and len(set(j2)) > 1:
            try:
                kappa = cohen_kappa_score(j1, j2)
            except Exception:
                kappa = float('nan')
            try:
                pearson, p_val = pearsonr(j1, j2)
            except Exception:
                pearson, p_val = float('nan'), float('nan')
        else:
            kappa = float('nan')
            pearson = float('nan')

        print(f"{d:<15} Kappa={kappa:.3f}  Pearson={pearson:.3f}")

    # Preference agreement
    j1_prefs = [r['judge1_preferred'] for r in results]
    j2_prefs = [r['judge2_preferred'] for r in results]
    pref_agree = sum(1 for a, b in zip(j1_prefs, j2_prefs) if a == b)
    print(f"\nPreference agreement: {pref_agree}/{len(results)} ({pref_agree/len(results)*100:.1f}%)")

    # Win rates
    for judge_key, judge_name in [('judge1_preferred', 'Judge 1'), ('judge2_preferred', 'Judge 2')]:
        prefs = [r[judge_key] for r in results]
        # Map back: if system_is_a and preferred is A, system wins
        sys_wins = sum(1 for r in results
                      if (r['system_is_a'] and r[judge_key] == 'A') or
                         (not r['system_is_a'] and r[judge_key] == 'B'))
        base_wins = sum(1 for r in results
                       if (r['system_is_a'] and r[judge_key] == 'B') or
                          (not r['system_is_a'] and r[judge_key] == 'A'))
        ties = len(results) - sys_wins - base_wins
        print(f"{judge_name}: System wins={sys_wins}, Baseline wins={base_wins}, Ties={ties}")

    # Save
    out_file = RESULTS_DIR / 'reply_eval_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
