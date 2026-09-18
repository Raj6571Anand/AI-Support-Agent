import json
import os
from pathlib import Path

# Paths
HUMAN_LABELS_PATH = Path(r"d:\Hiver\golden_set\human_labels.json")
RESULTS_PATH = Path(r"d:\Hiver\results\kappa_results.json")

def compute_kappa(y_true, y_pred, labels=None):
    if not labels:
        labels = list(set(y_true) | set(y_pred))
    
    n = len(y_true)
    if n == 0:
        return 0.0
        
    label_to_idx = {l: i for i, l in enumerate(labels)}
    mat = [[0] * len(labels) for _ in range(len(labels))]
    
    for t, p in zip(y_true, y_pred):
        mat[label_to_idx[t]][label_to_idx[p]] += 1
        
    po = sum(mat[i][i] for i in range(len(labels))) / n
    
    pe = 0
    for i in range(len(labels)):
        row_sum = sum(mat[i])
        col_sum = sum(mat[j][i] for j in range(len(labels)))
        pe += (row_sum / n) * (col_sum / n)
        
    if pe == 1:
        return 1.0
        
    return (po - pe) / (1 - pe)

def get_confusion_matrix(y_true, y_pred, labels):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    mat = [[0] * len(labels) for _ in range(len(labels))]
    
    for t, p in zip(y_true, y_pred):
        mat[label_to_idx[t]][label_to_idx[p]] += 1
        
    return mat

def main():
    if not HUMAN_LABELS_PATH.exists():
        print(f"File not found: {HUMAN_LABELS_PATH}")
        return
        
    with open(HUMAN_LABELS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data:
        print("No human labels found.")
        return
        
    # data is a dict: {id: {llm_intent, human_intent, ...}}
    items = list(data.values())
    
    # Intents
    llm_intents = [item['llm_intent'] for item in items]
    human_intents = [item['human_intent'] for item in items]
    intent_labels = sorted(list(set(llm_intents) | set(human_intents)))
    
    kappa_intent = compute_kappa(llm_intents, human_intents, intent_labels)
    
    # Escalation
    llm_esc = [item['llm_escalation'] for item in items]
    human_esc = [item['human_escalation'] for item in items]
    esc_labels = [True, False]
    kappa_esc = compute_kappa(llm_esc, human_esc, esc_labels)
    
    # Per-intent agreement rate (from LLM perspective)
    agreement = {}
    for intent in intent_labels:
        total = sum(1 for llm, human in zip(llm_intents, human_intents) if llm == intent)
        agreed = sum(1 for llm, human in zip(llm_intents, human_intents) if llm == intent and llm == human)
        rate = agreed / total if total > 0 else 0
        agreement[intent] = {"agreed": agreed, "total": total, "rate": rate}
        
    # Confusion matrix
    conf_mat = get_confusion_matrix(llm_intents, human_intents, intent_labels)
    conf_dict = {
        "labels": intent_labels,
        "matrix": conf_mat
    }
    
    # Ceiling
    noise_frac_intent = (1 - kappa_intent) / (1 + kappa_intent) if kappa_intent != -1 else 1.0
    ceiling_intent = 1 - noise_frac_intent
    
    noise_frac_esc = (1 - kappa_esc) / (1 + kappa_esc) if kappa_esc != -1 else 1.0
    ceiling_esc = 1 - noise_frac_esc
    
    results = {
        "kappa_intent": kappa_intent,
        "kappa_escalation": kappa_esc,
        "per_intent_agreement": agreement,
        "confusion_matrix": conf_dict,
        "implied_accuracy_ceiling_intent": ceiling_intent,
        "implied_accuracy_ceiling_escalation": ceiling_esc
    }
    
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print("=== COHEN'S KAPPA ANALYSIS ===")
    print(f"Total labeled items: {len(items)}")
    print(f"Intent Kappa:     {kappa_intent:.4f} (Ceiling: {ceiling_intent:.2%})")
    print(f"Escalation Kappa: {kappa_esc:.4f} (Ceiling: {ceiling_esc:.2%})")
    print("\n--- Per-Intent Agreement ---")
    for intent, stats in agreement.items():
        print(f"{intent:20s}: {stats['rate']:.2%} ({stats['agreed']}/{stats['total']})")
        
    print("\n--- Confusion Matrix (Row: LLM, Col: Human) ---")
    print(f"{'':20s} " + " ".join([f"{l[:3]:>4s}" for l in intent_labels]))
    for i, row in enumerate(conf_mat):
        print(f"{intent_labels[i]:20s} " + " ".join([f"{val:>4d}" for val in row]))
        
    print(f"\nResults saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
