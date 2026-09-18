import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import *
from src.pipeline.classifier import HybridClassifier
from src.pipeline import llm_cache

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    golden_set_path = Path(GOLDEN_SET_LABELLED)
    if not golden_set_path.exists():
        logger.error(f"Golden set not found at {golden_set_path}")
        return
        
    with open(golden_set_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)
        
    logger.info(f"Loaded {len(golden_set)} examples from golden set")
    
    classifier = HybridClassifier()
    
    y_true = []
    y_pred = []
    confusion_examples = defaultdict(list)
    support = defaultdict(int)
    
    # Run classification
    for item in golden_set:
        text = item['customer_text']
        true_intent = item.get('true_intent')
        if not true_intent:
            continue
            
        pred_result = classifier.classify(text)
        pred_intent = pred_result['intent']
        
        y_true.append(true_intent)
        y_pred.append(pred_intent)
        support[true_intent] += 1
        
        if true_intent != pred_intent:
            confusion_examples[(true_intent, pred_intent)].append(text)
            
    # Compute full confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
        
    intents = sorted(list(INTENTS.keys()))
    
    # Compute pairwise confusion
    flagged_pairs = []
    seen_pairs = set()
    
    for i in intents:
        for j in intents:
            if i == j: continue
            if (i, j) in seen_pairs or (j, i) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            
            i_as_j = matrix[i][j]
            j_as_i = matrix[j][i]
            
            support_i = support[i]
            support_j = support[j]
            
            if support_i == 0 and support_j == 0:
                continue
                
            rate_i = i_as_j / support_i if support_i else 0
            rate_j = j_as_i / support_j if support_j else 0
            sym_rate = (i_as_j + j_as_i) / (support_i + support_j)
            total_confused = i_as_j + j_as_i
            
            if total_confused > 0.1 * support_i or total_confused > 0.1 * support_j:
                # Get up to 3 actual examples of confusion
                exs = confusion_examples[(i, j)] + confusion_examples[(j, i)]
                
                flagged_pairs.append({
                    "pair": [i, j],
                    "i_as_j": i_as_j,
                    "j_as_i": j_as_i,
                    "rate_i": rate_i,
                    "rate_j": rate_j,
                    "symmetric_rate": sym_rate,
                    "total_confused": total_confused,
                    "support_i": support_i,
                    "support_j": support_j,
                    "examples": exs[:3]
                })
                
    report = {
        "confusion_matrix": {i: {j: matrix[i][j] for j in intents} for i in intents},
        "flagged_pairs": flagged_pairs,
        "recommendations": "Consider merging pairs with high symmetric confusion. For highly asymmetric confusion, clarify descriptions or add distinct examples to resolve boundary overlap."
    }
    
    out_path = RESULTS_DIR / "taxonomy_audit.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*50)
    print("Taxonomy Audit Report")
    print("="*50)
    
    print("\nFlagged Pairs (Symmetric confusion > 10% of support):")
    for fp in flagged_pairs:
        i, j = fp["pair"]
        print(f"\n- {i} & {j}")
        print(f"  {i} predicted as {j}: {fp['i_as_j']} / {fp['support_i']} ({fp['rate_i']:.2%})")
        print(f"  {j} predicted as {i}: {fp['j_as_i']} / {fp['support_j']} ({fp['rate_j']:.2%})")
        print(f"  Symmetric confusion: {fp['total_confused']} / {fp['support_i']+fp['support_j']} ({fp['symmetric_rate']:.2%})")
        print("  Examples:")
        for ex in fp['examples']:
            print(f"    * {ex.replace(chr(10), ' ')[:120]}...")
            
    print("\nGeneral Inquiry Analysis (F1=0.18):")
    gi = "general_inquiry"
    gi_confusions = [(other, matrix[gi][other], matrix[other][gi]) for other in intents if other != gi]
    gi_confusions.sort(key=lambda x: x[1] + x[2], reverse=True)
    for other, gi_as_other, other_as_gi in gi_confusions:
        if gi_as_other > 0 or other_as_gi > 0:
            print(f"  - vs {other}: True GI predicted as {other} = {gi_as_other} | True {other} predicted as GI = {other_as_gi}")
            
    print(f"\nSaved full report to {out_path}")

if __name__ == "__main__":
    main()
