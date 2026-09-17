import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.pipeline.classifier import KeywordClassifier, EmbeddingClassifier, HybridClassifier

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    scores = []
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        score = metric_fn(np.array(y_true)[indices], np.array(y_pred)[indices], average='macro', zero_division=0)
        scores.append(score)
    lower = np.percentile(scores, (1-ci)/2 * 100)
    upper = np.percentile(scores, (1+ci)/2 * 100)
    return np.mean(scores), lower, upper

class EmbeddingWrapper:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def classify(self, text):
        res = self.classifier.classify(text)
        if isinstance(res, dict):
            return res
        elif isinstance(res, tuple):
            return {'intent': res[0], 'confidence': res[1] if len(res) > 1 else 1.0}
        else:
            return {'intent': str(res), 'confidence': 1.0}

def main():
    try:
        with open(GOLDEN_SET_LABELLED, 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
    except Exception as e:
        print(f"Error loading golden set: {e}")
        return

    kw = KeywordClassifier()
    emb = EmbeddingWrapper(EmbeddingClassifier())
    hybrid = HybridClassifier()

    y_true = []
    preds_freq = []
    preds_kw = []
    preds_emb = []
    preds_hyb = []

    for item in tqdm(golden_set, desc="Evaluating Classifiers"):
        text = item.get('customer_text', '')
        true_intent = item.get('true_intent', 'unknown')
        
        y_true.append(true_intent)
        
        # MostFrequent baseline
        preds_freq.append('order_issue')
        
        try:
            res_kw = kw.classify(text)
            preds_kw.append(res_kw.get('intent', 'unknown'))
        except Exception:
            preds_kw.append('unknown')
            
        try:
            res_emb = emb.classify(text)
            preds_emb.append(res_emb.get('intent', 'unknown'))
        except Exception:
            preds_emb.append('unknown')
            
        try:
            res_hyb = hybrid.classify(text)
            preds_hyb.append(res_hyb.get('intent', 'unknown'))
        except Exception:
            preds_hyb.append('unknown')

    models = [
        ("Most Frequent", preds_freq),
        ("Keyword", preds_kw),
        ("Embedding", preds_emb),
        ("Hybrid", preds_hyb)
    ]

    print("\n| Classifier     | Accuracy | Macro F1 | Micro F1 | 95% CI          |")
    print("|----------------|----------|----------|----------|-----------------|")
    
    results = {}
    
    for name, preds in models:
        if not y_true:
            break
        acc = accuracy_score(y_true, preds)
        mac = f1_score(y_true, preds, average='macro', zero_division=0)
        mic = f1_score(y_true, preds, average='micro', zero_division=0)
        
        mean_mac, lower, upper = bootstrap_ci(y_true, preds, f1_score)
        
        print(f"| {name:<14} | {acc*100:>7.1f}% | {mac:>8.3f} | {mic:>8.3f} | ({lower:.3f} - {upper:.3f}) |")
        
        results[name] = {
            "accuracy": acc,
            "macro_f1": mac,
            "micro_f1": mic,
            "ci_lower": lower,
            "ci_upper": upper,
            "classification_report": classification_report(y_true, preds, zero_division=0, output_dict=True)
        }

    if y_true:
        print("\nHybrid Classifier Report:")
        print(classification_report(y_true, preds_hyb, zero_division=0))
        
        print("\nHybrid Confusion Matrix:")
        labels = sorted(list(set(y_true + preds_hyb)))
        cm = confusion_matrix(y_true, preds_hyb, labels=labels)
        
        print(f"{'':>15} " + " ".join([f"{str(l)[:5]:>5}" for l in labels]))
        for i, row in enumerate(cm):
            print(f"{str(labels[i])[:15]:>15} " + " ".join([f"{val:>5}" for val in row]))
            
    try:
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        with open(Path(RESULTS_DIR) / 'classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")
        
if __name__ == '__main__':
    main()
