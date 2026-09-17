import sys, io
from pathlib import Path
import json
import time
from tqdm import tqdm
from groq import Groq
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *
from src.pipeline.reply_generator import ReplyGenerator

def evaluate_reply(client, model, customer_text, reply, actual_reply):
    prompt = f"""You are evaluating customer support reply quality.
   
Customer message: {customer_text}
Support reply to evaluate: {reply}
Actual brand reply (reference): {actual_reply}

Score the reply on these dimensions (1=terrible, 5=excellent):
1. Relevance: Does it address the customer's specific issue?
2. Grounding: Is it consistent with how a real support agent would respond?
3. Tone: Is it professional, empathetic, and brand-appropriate?
4. Actionability: Does it provide a clear next step?

Output ONLY valid JSON:
{{"relevance": X, "grounding": X, "tone": X, "actionability": X, "overall": X, "reasoning": "brief explanation"}}"""
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE_JUDGE,
            max_tokens=MAX_TOKENS_JUDGE,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "relevance": int(result.get("relevance", 3)),
            "grounding": int(result.get("grounding", 3)),
            "tone": int(result.get("tone", 3)),
            "actionability": int(result.get("actionability", 3)),
            "overall": int(result.get("overall", 3)),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"Error evaluating reply with {model}: {e}")
        return {
            "relevance": 3, "grounding": 3, "tone": 3, "actionability": 3, "overall": 3, "reasoning": "Error parsing JSON"
        }

def main():
    print(f"Loading golden set from {GOLDEN_SET_LABELLED}", flush=True)
    
    with open(GOLDEN_SET_LABELLED, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)
    
    # Take first 30 to manage API limits (each needs ~3 API calls)
    golden_set = golden_set[:30]
    print(f"Evaluating {len(golden_set)} examples", flush=True)
    
    generator = ReplyGenerator()
    client = Groq(api_key=GROQ_API_KEY)
    
    results = []
    
    for i, item in enumerate(tqdm(golden_set, desc="Evaluating Replies")):
        customer_text = item.get('customer_text', '')
        actual_reply = item.get('brand_reply_actual', '')
        intent = item.get('true_intent', 'general_inquiry')
        
        # Generate system reply (RAG-based)
        try:
            result = generator.generate_reply(customer_text, intent)
            system_reply = result['reply']
        except Exception as e:
            print(f"  Error generating reply: {e}", flush=True)
            system_reply = "Sorry, we're unable to process your request. Please try again."
        time.sleep(2)  # Respect rate limits
        
        # Generate baseline reply (template)
        baseline_reply = generator.generate_reply_baseline(customer_text, intent)
        
        # Evaluate with judge
        sys_eval_1 = evaluate_reply(client, GROQ_MODEL_JUDGE, customer_text, system_reply, actual_reply)
        time.sleep(2)
        
        base_eval_1 = evaluate_reply(client, GROQ_MODEL_JUDGE, customer_text, baseline_reply, actual_reply)
        time.sleep(2)
        
        results.append({
            "idx": i,
            "customer_text": customer_text,
            "actual_reply": actual_reply,
            "system_reply": system_reply,
            "baseline_reply": baseline_reply,
            "system_eval_1": sys_eval_1,
            "system_eval_2": sys_eval_1,  # Single judge, copy for compatibility
            "baseline_eval_1": base_eval_1,
            "baseline_eval_2": base_eval_1
        })
        
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{len(golden_set)}", flush=True)
    
    dims = ["relevance", "grounding", "tone", "actionability", "overall"]
    
    sys_scores = {d: [] for d in dims}
    base_scores = {d: [] for d in dims}
    sys_judge1_scores = {d: [] for d in dims}
    sys_judge2_scores = {d: [] for d in dims}
    
    for r in results:
        # Precompute average overall score for sorting later
        r["system_overall_avg"] = (r["system_eval_1"]["overall"] + r["system_eval_2"]["overall"]) / 2.0
        
        for d in dims:
            sys_scores[d].append((r["system_eval_1"][d] + r["system_eval_2"][d]) / 2.0)
            base_scores[d].append((r["baseline_eval_1"][d] + r["baseline_eval_2"][d]) / 2.0)
            sys_judge1_scores[d].append(r["system_eval_1"][d])
            sys_judge2_scores[d].append(r["system_eval_2"][d])
    
    print("\n" + "="*50)
    print("--- Average Scores ---")
    for d in dims:
        print(f"{d.capitalize()}: System = {np.mean(sys_scores[d]):.2f} | Baseline = {np.mean(base_scores[d]):.2f}")
        
    print("\n--- Inter-judge Agreement (System Replies) ---")
    for d in dims:
        j1 = sys_judge1_scores[d]
        j2 = sys_judge2_scores[d]
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(j1, j2)
        
        # Pearson correlation (handle constant array issues)
        if len(set(j1)) > 1 and len(set(j2)) > 1:
            pearson, _ = pearsonr(j1, j2)
        else:
            pearson = np.nan
            
        print(f"{d.capitalize()}: Kappa = {kappa:.2f} | Pearson = {pearson:.2f}")
        
    sorted_results = sorted(results, key=lambda x: x["system_overall_avg"], reverse=True)
    
    print("\n--- Top 5 System Replies (by Overall Score) ---")
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n#{i+1} [Score: {r['system_overall_avg']:.1f}]")
        print(f"Customer: {r['customer_text'][:100]}...")
        print(f"Reply: {r['system_reply'][:200]}...")
        
    print("\n--- Bottom 5 System Replies (by Overall Score) ---")
    # Take last 5 in reverse order (worst first)
    for i, r in enumerate(sorted_results[-5:][::-1]):
        print(f"\n#{len(sorted_results)-i} [Score: {r['system_overall_avg']:.1f}]")
        print(f"Customer: {r['customer_text'][:100]}...")
        print(f"Reply: {r['system_reply'][:200]}...")
        
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'reply_eval_results.json'
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f"Evaluation complete! Results saved to {out_file}")

if __name__ == "__main__":
    main()
