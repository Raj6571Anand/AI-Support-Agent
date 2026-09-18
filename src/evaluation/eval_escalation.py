import sys, io
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

from src.pipeline.classifier import HybridClassifier
from src.pipeline.escalation import EscalationDecider

def bootstrap_f1(y_true, y_pred, n_iterations=1000, random_state=42):
    np.random.seed(random_state)
    n_size = len(y_true)
    stats = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for i in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        stats.append(f1_score(y_true_arr[indices], y_pred_arr[indices], zero_division=0))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def evaluate_escalation():
    with open(GOLDEN_SET_LABELLED, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)

    classifier = HybridClassifier()
    decider = EscalationDecider()

    y_true = []
    y_pred_sys = []
    y_pred_base = []

    false_positives = []
    false_negatives = []
    criteria_counter = Counter()

    for item in tqdm(golden_data, desc="Evaluating Escalation"):
        text = item.get("customer_text", "")
        true_esc = item.get("true_escalation", False)
        thread_ctx = item.get("thread_context", [])
        
        cls_result = classifier.classify(text)
        intent = cls_result['intent']
        confidence = cls_result['confidence']
        
        sys_decision = decider.decide(text, intent, confidence, thread_context=thread_ctx)
        base_decision = decider.decide_baseline(text, intent)
        
        pred_esc = (sys_decision.get("decision") == "escalate")
        base_esc = (base_decision.get("decision") == "escalate")
        
        y_true.append(true_esc)
        y_pred_sys.append(pred_esc)
        y_pred_base.append(base_esc)
        
        if pred_esc:
            for criterion in sys_decision.get("criteria_triggered", []):
                criteria_counter[criterion] += 1
                
        if pred_esc and not true_esc:
            false_positives.append({
                "text": text,
                "reason": sys_decision.get("reason", ""),
                "criteria": sys_decision.get("criteria_triggered", [])
            })
        elif not pred_esc and true_esc:
            false_negatives.append({
                "text": text,
                "reason": sys_decision.get("reason", ""),
                "criteria": sys_decision.get("criteria_triggered", [])
            })

    # Add baselines
    y_pred_always = [True] * len(y_true)
    y_pred_never = [False] * len(y_true)
    
    # Random baseline: probability of True = base rate
    np.random.seed(42)
    p_true = sum(y_true) / len(y_true)
    y_pred_random = np.random.choice([True, False], p=[p_true, 1-p_true], size=len(y_true)).tolist()

    systems = {
        "System": y_pred_sys,
        "Baseline": y_pred_base,
        "Always-escalate": y_pred_always,
        "Never-escalate": y_pred_never,
        "Random": y_pred_random
    }
    
    results = {}
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'F1 95% CI'}")
    print("-" * 80)
    
    for name, preds in systems.items():
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, pos_label=True, zero_division=0)
        rec = recall_score(y_true, preds, pos_label=True, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=True, zero_division=0)
        
        ci_lower, ci_upper = bootstrap_f1(y_true, preds)
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "f1_ci_lower": float(ci_lower),
            "f1_ci_upper": float(ci_upper)
        }
        
        print(f"{name:<20} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}]")

    print("=" * 80)
    
    print("\nCRITERIA DISTRIBUTION:")
    for criterion, count in criteria_counter.most_common():
        print(f"- {criterion}: {count}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / 'escalation_results.json'
    
    output_data = {
        "metrics": results,
        "criteria_distribution": dict(criteria_counter)
    }
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == '__main__':
    evaluate_escalation()
