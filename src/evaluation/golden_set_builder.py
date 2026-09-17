"""
Golden Evaluation Set Builder — Fast Version
Uses keyword pre-classification for stratification (instant),
then LLM for labeling only the final 200 examples.
"""
import sys
import json
import re
import random
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

from groq import Groq


KEYWORD_MAP = {
    "order_issue": ['order', 'deliver', 'shipping', 'package', 'tracking', 'arrived', 'shipped', 'dispatch'],
    "refund_return": ['refund', 'return', 'money back', 'charged twice', 'overcharged', 'reimburse'],
    "account_access": ['account', 'password', 'login', 'sign in', 'locked', 'verify', 'hacked', 'log in'],
    "subscription": ['prime', 'membership', 'subscription', 'renewal', 'subscribe'],
    "product_issue": ['broken', 'defective', 'not working', 'damaged', 'quality', 'kindle', 'alexa', 'echo', 'fire tablet'],
    "payment_billing": ['payment', 'gift card', 'credit card', 'charge', 'transaction', 'promo', 'coupon'],
    "feedback": ['thank', 'thanks', 'great job', 'worst', 'terrible', 'horrible', 'love', 'amazing', 'awesome', 'appreciate'],
    "general_inquiry": ['how do', 'what is', 'policy', 'can i', 'where is', 'how long', 'do you'],
}


def keyword_classify(text):
    """Fast keyword-based intent classification for stratification."""
    text_lower = text.lower()
    counts = {}
    for intent, kws in KEYWORD_MAP.items():
        counts[intent] = sum(1 for kw in kws if kw in text_lower)
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "general_inquiry"
    return best


def label_with_llm(client, customer_text, thread_context=None):
    """Use LLM to label a single example."""
    intent_list = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in INTENTS.items()
    ])

    context_str = ""
    if thread_context and len(thread_context) > 0:
        context_str = "\nPrevious messages:\n"
        for msg in thread_context[:3]:  # Limit context
            context_str += f"  [{msg['role']}]: {msg['text'][:100]}\n"

    prompt = f"""Classify this customer support message into exactly ONE intent:
{intent_list}

{context_str}
Message: "{customer_text[:200]}"

Also: should this be escalated to a human? (true/false)

Output ONLY valid JSON: {{"intent": "name", "escalation": true/false, "reasoning": "brief"}}
/no_think"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_GENERATION,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        intent = result.get("intent", "general_inquiry")
        if intent not in INTENT_NAMES:
            intent = "general_inquiry"
        return {
            "intent": intent,
            "escalation": bool(result.get("escalation", False)),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        return {"intent": "general_inquiry", "escalation": False, "reasoning": f"Error: {e}"}


def main():
    print("=" * 70, flush=True)
    print("GOLDEN EVALUATION SET BUILDER (Fast)", flush=True)
    print("=" * 70, flush=True)

    # Load processed data
    print("\n[1/4] Loading processed data...", flush=True)
    with open(PROCESSED_DATA, 'r', encoding='utf-8') as f:
        all_pairs = json.load(f)
    print(f"  Total pairs available: {len(all_pairs)}", flush=True)

    # Step 1: Keyword-based pre-classification (instant, no API calls)
    print("\n[2/4] Keyword pre-classification for stratification...", flush=True)
    for pair in all_pairs:
        pair['_keyword_intent'] = keyword_classify(pair['customer_text'])

    kw_counts = Counter(p['_keyword_intent'] for p in all_pairs)
    print("  Keyword intent distribution:", flush=True)
    for intent, count in sorted(kw_counts.items(), key=lambda x: -x[1]):
        print(f"    {intent:25s}: {count}", flush=True)

    # Step 2: Stratified sampling — ~25 per intent
    print(f"\n[3/4] Stratified sampling (target: {EXAMPLES_PER_INTENT} per intent)...", flush=True)
    random.seed(42)

    intent_groups = {}
    for pair in all_pairs:
        intent = pair['_keyword_intent']
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(pair)

    selected_pairs = []
    for intent_name in INTENT_NAMES:
        available = intent_groups.get(intent_name, [])
        random.shuffle(available)
        target = min(EXAMPLES_PER_INTENT, len(available))
        selected = available[:target]
        selected_pairs.extend(selected)
        print(f"  {intent_name:25s}: {len(selected)} selected (from {len(available)} available)", flush=True)

    # Add edge cases
    # Short messages
    short = [p for p in all_pairs if len(p['customer_text']) < 30 and p not in selected_pairs]
    random.shuffle(short)
    selected_pairs.extend(short[:10])
    # Frustrated messages
    frust_words = ['worst', 'terrible', 'horrible', 'never again', 'scam', 'unacceptable', 'furious']
    frustrated = [p for p in all_pairs if any(w in p['customer_text'].lower() for w in frust_words) and p not in selected_pairs]
    random.shuffle(frustrated)
    selected_pairs.extend(frustrated[:10])

    print(f"\n  Total selected for labeling: {len(selected_pairs)}", flush=True)

    # Step 3: LLM labeling of only the selected examples
    print(f"\n[4/4] LLM labeling {len(selected_pairs)} examples...", flush=True)
    client = Groq(api_key=GROQ_API_KEY)

    golden_examples = []
    errors = 0
    for i, pair in enumerate(selected_pairs):
        if i % 25 == 0:
            print(f"  Labeling {i}/{len(selected_pairs)}...", flush=True)

        label = label_with_llm(client, pair['customer_text'], pair.get('thread_context'))

        if "Error" in label.get('reasoning', ''):
            errors += 1
            # Fall back to keyword classification
            label['intent'] = pair['_keyword_intent']
            label['escalation'] = pair.get('escalation_signals', {}).get('is_escalation', False)
            label['reasoning'] = f"LLM error, fell back to keyword ({pair['_keyword_intent']})"

        golden_example = {
            'id': pair['id'],
            'customer_text': pair['customer_text'],
            'customer_text_raw': pair.get('customer_text_raw', pair['customer_text']),
            'brand_reply_actual': pair['brand_reply'],
            'thread_context': pair.get('thread_context', []),
            'true_intent': label['intent'],
            'true_escalation': label['escalation'],
            'labeling_notes': label['reasoning'],
            'label_source': 'llm_assisted' if 'Error' not in label.get('reasoning', '') else 'keyword_fallback',
            'thread_length': pair.get('thread_length', 1),
            'escalation_signals': pair.get('escalation_signals', {}),
        }
        golden_examples.append(golden_example)
        time.sleep(0.5)  # Rate limiting

    # Deduplicate
    seen_ids = set()
    unique = []
    for ex in golden_examples:
        if ex['id'] not in seen_ids:
            seen_ids.add(ex['id'])
            unique.append(ex)
    golden_examples = unique

    # Save
    print(f"\n  Saving {len(golden_examples)} examples to {GOLDEN_SET_LABELLED}...", flush=True)
    with open(GOLDEN_SET_LABELLED, 'w', encoding='utf-8') as f:
        json.dump(golden_examples, f, indent=2, ensure_ascii=False)

    # Stats
    final_counts = Counter(ex['true_intent'] for ex in golden_examples)
    esc_counts = Counter(ex['true_escalation'] for ex in golden_examples)
    print(f"\n  Final intent distribution:", flush=True)
    for intent, count in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"    {intent:25s}: {count}", flush=True)
    print(f"\n  Escalation: escalate={esc_counts.get(True, 0)}, auto_handle={esc_counts.get(False, 0)}", flush=True)
    print(f"  LLM errors (keyword fallback): {errors}", flush=True)
    print("\n  Golden set complete!", flush=True)


if __name__ == "__main__":
    main()
