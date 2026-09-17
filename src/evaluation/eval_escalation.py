import sys, io
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

from src.pipeline.classifier import HybridClassifier
from src.pipeline.escalation import EscalationDecider

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
        
        # Classifier returns dict: {intent, confidence, method, scores}
        cls_result = classifier.classify(text)
        intent = cls_result['intent']
        confidence = cls_result['confidence']
        
        sys_decision = decider.decide(text, intent, confidence, thread_context=thread_ctx)
        base_decision = decider.decide_baseline(text, intent)
        
        # Decision key is 'decision': 'escalate' or 'auto_handle'
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

    print("\n" + "="*50)
    print("SYSTEM PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_sys):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_sys, pos_label=True, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred_sys, pos_label=True, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred_sys, pos_label=True, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_sys, zero_division=0))

    print("\n" + "="*50)
    print("BASELINE PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_base):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_base, pos_label=True, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred_base, pos_label=True, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred_base, pos_label=True, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_base, zero_division=0))

    print("\n" + "="*50)
    print("CRITERIA DISTRIBUTION")
    print("="*50)
    for criterion, count in criteria_counter.most_common():
        print(f"- {criterion}: {count}")

    print("\n" + "="*50)
    print("TOP 5 FALSE POSITIVES (Unnecessary Escalation)")
    print("="*50)
    for i, fp in enumerate(false_positives[:5]):
        print(f"{i+1}. Text: {fp['text'][:150]}")
        print(f"   Reason: {fp['reason']}")
        print(f"   Criteria: {fp['criteria']}\n")

    print("\n" + "="*50)
    print("TOP 5 FALSE NEGATIVES (Missed Escalation)")
    print("="*50)
    for i, fn in enumerate(false_negatives[:5]):
        print(f"{i+1}. Text: {fn['text'][:150]}")
        print(f"   Reason: {fn['reason']}")
        print(f"   Criteria: {fn['criteria']}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "system": {
            "accuracy": accuracy_score(y_true, y_pred_sys),
            "precision": precision_score(y_true, y_pred_sys, pos_label=True, zero_division=0),
            "recall": recall_score(y_true, y_pred_sys, pos_label=True, zero_division=0),
            "f1": f1_score(y_true, y_pred_sys, pos_label=True, zero_division=0)
        },
        "baseline": {
            "accuracy": accuracy_score(y_true, y_pred_base),
            "precision": precision_score(y_true, y_pred_base, pos_label=True, zero_division=0),
            "recall": recall_score(y_true, y_pred_base, pos_label=True, zero_division=0),
            "f1": f1_score(y_true, y_pred_base, pos_label=True, zero_division=0)
        },
        "criteria_distribution": dict(criteria_counter)
    }

    out_file = RESULTS_DIR / 'escalation_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    evaluate_escalation()
