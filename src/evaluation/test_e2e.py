"""
End-to-End Pipeline Test (Task 12)
Tests app.py's process_message() against 20 representative inputs.
Documents what works and what doesn't.
"""
import sys, io, json, time
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

# Import the pipeline function directly (not the Gradio UI)
from src.demo.app import process_message

TEST_CASES = [
    # Order issues
    {"text": "My package hasn't arrived and it's been 5 days past the delivery date", "expected_intent": "order_issue", "expected_escalation": True},
    {"text": "Where is my order? The tracking hasn't updated in 3 days", "expected_intent": "order_issue", "expected_escalation": True},
    # Refund/return
    {"text": "I want a refund for this defective product", "expected_intent": "refund_return", "expected_escalation": True},
    {"text": "How do I return an item I purchased last week?", "expected_intent": "refund_return", "expected_escalation": False},
    # Account access
    {"text": "I can't sign into my Amazon account", "expected_intent": "account_access", "expected_escalation": True},
    {"text": "Someone hacked my account and changed my email", "expected_intent": "account_access", "expected_escalation": True},
    # Subscription
    {"text": "How do I cancel my Prime membership?", "expected_intent": "subscription", "expected_escalation": False},
    {"text": "I was charged for Prime without consent", "expected_intent": "subscription", "expected_escalation": True},
    # Product issues
    {"text": "My Kindle won't turn on anymore", "expected_intent": "product_issue", "expected_escalation": True},
    {"text": "The item I received is completely different from the listing", "expected_intent": "product_issue", "expected_escalation": True},
    # Payment/billing
    {"text": "My gift card balance disappeared from my account", "expected_intent": "payment_billing", "expected_escalation": True},
    {"text": "I see a charge from Amazon I don't recognize", "expected_intent": "payment_billing", "expected_escalation": True},
    # Feedback
    {"text": "Thanks for the quick help!", "expected_intent": "feedback", "expected_escalation": False},
    {"text": "Your customer service is terrible", "expected_intent": "feedback", "expected_escalation": False},
    # General inquiry
    {"text": "What is your return policy?", "expected_intent": "general_inquiry", "expected_escalation": False},
    {"text": "How long does standard shipping take?", "expected_intent": "general_inquiry", "expected_escalation": False},
    # Edge cases
    {"text": "I'm talking to my lawyer about this fraud", "expected_intent": None, "expected_escalation": True},
    {"text": "!!!", "expected_intent": None, "expected_escalation": False},
    {"text": "Can someone help me I've been waiting for hours and nobody is responding this is the worst experience ever", "expected_intent": None, "expected_escalation": True},
    {"text": "hi", "expected_intent": None, "expected_escalation": False},
]


def main():
    print("=" * 70)
    print("END-TO-END PIPELINE TEST")
    print("=" * 70)

    results = []
    intent_correct = 0
    intent_total = 0
    escalation_correct = 0
    total = len(TEST_CASES)
    errors = 0
    empty_replies = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i}/{total} ---")
        print(f"  Input: {tc['text'][:100]}")

        start = time.time()
        try:
            intent_out, conf_out, reply_out, esc_out, retrieved_out = process_message(tc['text'])
            latency = time.time() - start
            error = None
        except Exception as e:
            latency = time.time() - start
            intent_out, conf_out, reply_out, esc_out, retrieved_out = "", "", "", "", ""
            error = str(e)
            errors += 1
            print(f"  ERROR: {e}")

        # Parse outputs
        predicted_intent = ""
        if intent_out and "**" in intent_out:
            try:
                predicted_intent = intent_out.split("**")[1].lower().replace(" ", "_")
            except Exception:
                predicted_intent = "parse_error"

        is_escalation = "ESCALATE" in esc_out.upper() if esc_out else None
        reply_empty = not reply_out or reply_out.strip() == "" or reply_out.startswith("Please enter")

        if reply_empty:
            empty_replies += 1

        # Check intent accuracy (only where expected is set)
        intent_match = None
        if tc['expected_intent']:
            intent_match = predicted_intent == tc['expected_intent']
            intent_total += 1
            if intent_match:
                intent_correct += 1

        # Check escalation
        esc_match = None
        if is_escalation is not None:
            esc_match = is_escalation == tc['expected_escalation']
            if esc_match:
                escalation_correct += 1

        print(f"  Intent: {predicted_intent} (expected: {tc['expected_intent']}) {'✓' if intent_match else '✗' if intent_match is False else '—'}")
        print(f"  Escalation: {is_escalation} (expected: {tc['expected_escalation']}) {'✓' if esc_match else '✗' if esc_match is False else '—'}")
        print(f"  Reply empty: {reply_empty}")
        print(f"  Reply: {reply_out[:150] if reply_out else '(empty)'}")
        print(f"  Latency: {latency:.1f}s")

        results.append({
            "idx": i,
            "input": tc['text'],
            "expected_intent": tc['expected_intent'],
            "predicted_intent": predicted_intent,
            "intent_correct": intent_match,
            "expected_escalation": tc['expected_escalation'],
            "predicted_escalation": is_escalation,
            "escalation_correct": esc_match,
            "reply": reply_out,
            "reply_empty": reply_empty,
            "latency": latency,
            "error": error,
        })

        time.sleep(1)  # Rate limiting

    # Summary
    print(f"\n{'=' * 70}")
    print("E2E TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total tests: {total}")
    print(f"Errors: {errors}")
    print(f"Empty replies: {empty_replies}/{total} ({empty_replies/total*100:.0f}%)")
    if intent_total > 0:
        print(f"Intent accuracy: {intent_correct}/{intent_total} ({intent_correct/intent_total*100:.0f}%)")
    print(f"Escalation accuracy: {escalation_correct}/{total} ({escalation_correct/total*100:.0f}%)")
    avg_latency = np.mean([r['latency'] for r in results])
    print(f"Average latency: {avg_latency:.1f}s")

    # What works
    print(f"\n--- WHAT WORKS ---")
    working = [r for r in results if not r['error'] and not r['reply_empty']]
    print(f"  Successful end-to-end: {len(working)}/{total}")
    if working:
        intent_ok = [r for r in working if r['intent_correct'] is True]
        print(f"  Correct intent (of testable): {len(intent_ok)}")

    # What doesn't
    print(f"\n--- WHAT DOESN'T WORK ---")
    broken = [r for r in results if r['error'] or r['reply_empty']]
    for r in broken:
        print(f"  Test {r['idx']}: {'ERROR: ' + r['error'] if r['error'] else 'Empty reply'}")
        print(f"    Input: {r['input'][:80]}")

    # Save
    out_file = RESULTS_DIR / 'e2e_test_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
