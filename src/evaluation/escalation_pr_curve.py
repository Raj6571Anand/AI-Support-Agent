"""
Escalation Precision-Recall Curve + Operating Point Selection (Tasks 4, 5, 6)
- Sweeps threshold from 0.0 to 1.0
- Computes precision, recall, F1 at each threshold
- Produces PR curve plot
- Selects operating point using FN:FP cost ratio of 3:1
- Reports recall@precision=0.80
- Shows 20 worst false negatives with criterion analysis
"""
import sys
import io
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *
from src.pipeline.classifier import HybridClassifier
from src.pipeline.escalation import EscalationDecider

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, skipping plot generation")


def load_golden_set():
    with open(GOLDEN_SET_LABELLED, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_scores(golden_data, classifier, decider, aggregation='max'):
    """Run classifier + scored escalation on every golden set item."""
    records = []
    for item in golden_data:
        text = item.get("customer_text", "")
        true_esc = item.get("true_escalation", False)
        thread_ctx = item.get("thread_context", [])

        cls_result = classifier.classify(text)
        intent = cls_result['intent']
        confidence = cls_result['confidence']

        # Get scored result with a very low threshold so we always get a score
        scored = decider.decide_scored(
            text, intent, confidence,
            thread_context=thread_ctx,
            threshold=0.0,  # threshold doesn't matter for score computation
            aggregation=aggregation
        )

        records.append({
            'id': item.get('id', ''),
            'customer_text': text,
            'true_escalation': true_esc,
            'escalation_score': scored['escalation_score'],
            'criterion_scores': scored['criterion_scores'],
            'top_criterion': scored['top_criterion'],
            'intent': intent,
            'confidence': confidence,
            'thread_context': thread_ctx,
        })
    return records


def sweep_thresholds(records, thresholds=None):
    """Compute precision, recall, F1 at each threshold."""
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    y_true = [r['true_escalation'] for r in records]
    scores = [r['escalation_score'] for r in records]

    results = []
    for t in thresholds:
        y_pred = [s >= t for s in scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true)

        results.append({
            'threshold': float(t),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
    return results


def find_recall_at_precision(pr_results, target_precision=0.80):
    """Find the best recall achievable at or above the target precision."""
    candidates = [r for r in pr_results if r['precision'] >= target_precision]
    if not candidates:
        # Find the closest precision below target
        below = [r for r in pr_results if r['precision'] > 0]
        if below:
            best = max(below, key=lambda r: r['precision'])
            return best, False
        return None, False
    best = max(candidates, key=lambda r: r['recall'])
    return best, True


def find_cost_optimal(pr_results, fn_cost=3.0, fp_cost=1.0):
    """Find the threshold that minimizes total cost = fn_cost*FN + fp_cost*FP."""
    best = None
    best_cost = float('inf')
    for r in pr_results:
        cost = fn_cost * r['fn'] + fp_cost * r['fp']
        if cost < best_cost:
            best_cost = cost
            best = r
            best['cost'] = cost
    return best, best_cost


def get_false_negatives(records, threshold):
    """Get all false negatives at the given threshold, sorted by score ascending."""
    fns = []
    for r in records:
        pred_esc = r['escalation_score'] >= threshold
        if r['true_escalation'] and not pred_esc:
            fns.append(r)
    fns.sort(key=lambda x: x['escalation_score'])
    return fns


def plot_pr_curve(pr_results, cost_optimal, recall_at_p80, output_path):
    """Generate and save the PR curve plot."""
    if not HAS_MATPLOTLIB:
        return

    precisions = [r['precision'] for r in pr_results]
    recalls = [r['recall'] for r in pr_results]
    f1s = [r['f1'] for r in pr_results]
    thresholds = [r['threshold'] for r in pr_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PR Curve
    ax1.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
    if cost_optimal:
        ax1.plot(cost_optimal['recall'], cost_optimal['precision'], 'r*',
                markersize=15, label=f"Cost-optimal (t={cost_optimal['threshold']:.2f})")
    if recall_at_p80:
        ax1.plot(recall_at_p80['recall'], recall_at_p80['precision'], 'g^',
                markersize=12, label=f"Recall@P=0.80 (t={recall_at_p80['threshold']:.2f})")
    ax1.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, label='Precision=0.80')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # F1 vs Threshold
    ax2.plot(thresholds, f1s, 'g-', linewidth=2, label='F1 Score')
    ax2.plot(thresholds, precisions, 'b--', linewidth=1, alpha=0.7, label='Precision')
    ax2.plot(thresholds, recalls, 'r--', linewidth=1, alpha=0.7, label='Recall')
    if cost_optimal:
        ax2.axvline(x=cost_optimal['threshold'], color='red', linestyle=':',
                    alpha=0.7, label=f"Cost-optimal t={cost_optimal['threshold']:.2f}")
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Metrics vs Threshold', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to {output_path}")


def main():
    print("=" * 70)
    print("ESCALATION PR CURVE + OPERATING POINT ANALYSIS")
    print("=" * 70)

    golden_data = load_golden_set()
    print(f"Loaded {len(golden_data)} golden set examples")
    true_esc_count = sum(1 for g in golden_data if g.get('true_escalation'))
    print(f"True escalations: {true_esc_count}/{len(golden_data)} ({true_esc_count/len(golden_data)*100:.1f}%)")

    print("\nLoading classifier...")
    classifier = HybridClassifier()
    decider = EscalationDecider()

    # Test both aggregation modes
    for aggregation in ['max', 'weighted_mean']:
        print(f"\n{'=' * 70}")
        print(f"AGGREGATION: {aggregation}")
        print(f"{'=' * 70}")

        print("Computing escalation scores...")
        records = compute_scores(golden_data, classifier, decider, aggregation)

        # Score distribution
        scores = [r['escalation_score'] for r in records]
        print(f"\nScore distribution: min={min(scores):.3f}, max={max(scores):.3f}, "
              f"mean={np.mean(scores):.3f}, median={np.median(scores):.3f}")

        # Threshold sweep
        pr_results = sweep_thresholds(records)

        # Task 5: Recall@precision=0.80
        recall_at_p80, achieved = find_recall_at_precision(pr_results, 0.80)
        print(f"\n--- Recall@Precision=0.80 ---")
        if achieved and recall_at_p80:
            print(f"  Threshold: {recall_at_p80['threshold']:.2f}")
            print(f"  Precision: {recall_at_p80['precision']:.3f}")
            print(f"  Recall:    {recall_at_p80['recall']:.3f}")
            print(f"  F1:        {recall_at_p80['f1']:.3f}")
            print(f"  TP={recall_at_p80['tp']} FP={recall_at_p80['fp']} FN={recall_at_p80['fn']} TN={recall_at_p80['tn']}")
        else:
            print(f"  Precision=0.80 not achievable. Best available:")
            if recall_at_p80:
                print(f"  Precision={recall_at_p80['precision']:.3f} at threshold={recall_at_p80['threshold']:.2f}")

        # Task 5: Cost-optimal operating point (FN:FP = 3:1)
        FN_COST = 3.0
        FP_COST = 1.0
        cost_optimal, total_cost = find_cost_optimal(pr_results, FN_COST, FP_COST)
        print(f"\n--- Cost-Optimal Operating Point (FN:FP = {FN_COST}:{FP_COST}) ---")
        print(f"  Justification: Missing an escalation that needs human help is 3x more costly")
        print(f"  than unnecessarily escalating (wasting agent time vs. customer harm).")
        if cost_optimal:
            print(f"  Threshold: {cost_optimal['threshold']:.2f}")
            print(f"  Precision: {cost_optimal['precision']:.3f}")
            print(f"  Recall:    {cost_optimal['recall']:.3f}")
            print(f"  F1:        {cost_optimal['f1']:.3f}")
            print(f"  Cost:      {total_cost:.0f} (= {FN_COST}×{cost_optimal['fn']}FN + {FP_COST}×{cost_optimal['fp']}FP)")
            print(f"  TP={cost_optimal['tp']} FP={cost_optimal['fp']} FN={cost_optimal['fn']} TN={cost_optimal['tn']}")

        # Best F1
        best_f1 = max(pr_results, key=lambda r: r['f1'])
        print(f"\n--- Best F1 Operating Point ---")
        print(f"  Threshold: {best_f1['threshold']:.2f}")
        print(f"  F1:        {best_f1['f1']:.3f}")
        print(f"  Precision: {best_f1['precision']:.3f}")
        print(f"  Recall:    {best_f1['recall']:.3f}")

        # Compare with old system
        print(f"\n--- Comparison with Original System ---")
        print(f"  Old system F1:    0.439 (precision=0.661, recall=0.328)")
        print(f"  Old baseline F1:  0.682 (precision=0.669, recall=0.696)")
        if cost_optimal:
            print(f"  New scored F1:    {cost_optimal['f1']:.3f} (precision={cost_optimal['precision']:.3f}, recall={cost_optimal['recall']:.3f})")
            improvement = cost_optimal['f1'] - 0.439
            print(f"  Improvement over old system: {improvement:+.3f} F1")
            vs_baseline = cost_optimal['f1'] - 0.682
            print(f"  vs old baseline: {vs_baseline:+.3f} F1")

        # Plot
        if HAS_MATPLOTLIB:
            plot_path = RESULTS_DIR / f'escalation_pr_curve_{aggregation}.png'
            plot_pr_curve(pr_results, cost_optimal, recall_at_p80, plot_path)

        # Task 6: 20 worst false negatives at cost-optimal threshold
        if cost_optimal:
            chosen_threshold = cost_optimal['threshold']
            fns = get_false_negatives(records, chosen_threshold)
            print(f"\n{'=' * 70}")
            print(f"20 WORST FALSE NEGATIVES (threshold={chosen_threshold:.2f}, aggregation={aggregation})")
            print(f"Total false negatives: {len(fns)}")
            print(f"{'=' * 70}")
            for i, fn in enumerate(fns[:20], 1):
                print(f"\n--- FN #{i} (score={fn['escalation_score']:.3f}) ---")
                print(f"  Text: {fn['customer_text'][:200]}")
                print(f"  Intent: {fn['intent']} (confidence={fn['confidence']:.3f})")
                print(f"  Top criterion: {fn['top_criterion']} = {fn['criterion_scores'][fn['top_criterion']]:.3f}")
                print(f"  All criteria: ", end='')
                sorted_criteria = sorted(fn['criterion_scores'].items(), key=lambda x: x[1], reverse=True)
                print(', '.join([f"{k}={v:.2f}" for k, v in sorted_criteria if v > 0]))

            # Analyze patterns in false negatives
            print(f"\n--- FN Pattern Analysis ---")
            top_criteria = Counter(fn['top_criterion'] for fn in fns)
            print("Most common top criterion in FNs:")
            for criterion, count in top_criteria.most_common():
                print(f"  {criterion}: {count} ({count/len(fns)*100:.1f}%)")

            intent_dist = Counter(fn['intent'] for fn in fns)
            print("FN intent distribution:")
            for intent, count in intent_dist.most_common():
                print(f"  {intent}: {count}")

    # Save results
    save_results = {
        'golden_set_size': len(golden_data),
        'true_escalation_rate': true_esc_count / len(golden_data),
        'aggregation_results': {}
    }

    for aggregation in ['max', 'weighted_mean']:
        records = compute_scores(golden_data, classifier, decider, aggregation)
        pr_results = sweep_thresholds(records)
        recall_at_p80, _ = find_recall_at_precision(pr_results, 0.80)
        cost_optimal, total_cost = find_cost_optimal(pr_results, 3.0, 1.0)
        best_f1 = max(pr_results, key=lambda r: r['f1'])

        save_results['aggregation_results'][aggregation] = {
            'recall_at_p80': recall_at_p80,
            'cost_optimal': cost_optimal,
            'best_f1': best_f1,
            'fn_cost_ratio': '3:1',
        }

    out_file = RESULTS_DIR / 'escalation_pr_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
