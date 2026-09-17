"""
Data Preprocessing Pipeline
- Loads raw CSV, filters to AmazonHelp
- Filters to English messages
- Cleans text (normalizes @mentions, strips noise)
- Reconstructs conversation threads
- Extracts (customer_message, brand_reply, thread_context) triples
- Saves processed subsample as JSON
"""
import pandas as pd
import numpy as np
import json
import re
import sys
import io
from pathlib import Path
from tqdm import tqdm

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import *


def is_english(text: str) -> bool:
    """Simple English detection: check ASCII ratio + common English words."""
    if not isinstance(text, str) or len(text) < 10:
        return False
    # ASCII ratio (English text is mostly ASCII)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.8:
        return False
    # Check for at least one common English word
    common_words = {'the', 'is', 'my', 'i', 'you', 'to', 'and', 'a', 'have', 'it',
                    'for', 'not', 'was', 'but', 'can', 'do', 'this', 'that', 'with',
                    'are', 'on', 'be', 'from', 'or', 'an', 'will', 'has', 'help',
                    'your', 'we', 'me', 'what', 'how', 'why', 'when', 'please',
                    'need', 'get', 'got', 'been', 'just', 'no', 'so', 'if', 'up'}
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    overlap = words & common_words
    return len(overlap) >= 2


def clean_text(text: str) -> str:
    """Clean tweet text for processing."""
    if not isinstance(text, str):
        return ""
    # Replace specific @mentions with @user (keep @AmazonHelp for context)
    text = re.sub(r'@\d+', '@user', text)  # Anonymized user IDs like @115712
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def reconstruct_threads(df: pd.DataFrame) -> list[dict]:
    """
    Reconstruct conversation threads from the flat tweet dataframe.
    Returns list of conversation dicts with ordered messages.
    """
    # Build lookup dicts
    tweet_lookup = {}
    for _, row in df.iterrows():
        tweet_lookup[str(row['tweet_id'])] = row

    # Build child -> parent mapping
    children = {}  # parent_id -> list of child tweet_ids
    for _, row in df.iterrows():
        parent_id = row.get('in_response_to_tweet_id')
        if pd.notna(parent_id) and str(parent_id) != 'nan':
            parent_id = str(parent_id)
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(str(row['tweet_id']))

    # Find thread roots (messages with no parent in our dataset)
    root_ids = set()
    for _, row in df.iterrows():
        parent_id = row.get('in_response_to_tweet_id')
        if pd.isna(parent_id) or str(parent_id) == 'nan' or str(parent_id) not in tweet_lookup:
            root_ids.add(str(row['tweet_id']))

    # BFS from each root to build threads
    threads = []
    visited = set()

    for root_id in root_ids:
        if root_id in visited:
            continue

        thread_messages = []
        queue = [root_id]
        while queue:
            current_id = queue.pop(0)
            if current_id in visited or current_id not in tweet_lookup:
                continue
            visited.add(current_id)

            row = tweet_lookup[current_id]
            thread_messages.append({
                'tweet_id': str(row['tweet_id']),
                'author_id': str(row['author_id']),
                'is_customer': str(row['inbound']).lower() == 'true',
                'text': str(row['text']),
                'text_clean': clean_text(str(row['text'])),
                'created_at': str(row.get('created_at', '')),
            })

            # Add children to queue
            if current_id in children:
                for child_id in children[current_id]:
                    if child_id not in visited:
                        queue.append(child_id)

        if len(thread_messages) >= 2:  # At least one exchange
            threads.append(thread_messages)

    return threads


def extract_conversation_pairs(threads: list[list[dict]]) -> list[dict]:
    """
    From threads, extract (customer_message, brand_reply, thread_context) triples.
    These are the core training/retrieval units.
    """
    pairs = []
    for thread in threads:
        for i, msg in enumerate(thread):
            if msg['is_customer']:
                # Look for the brand reply to this customer message
                for j in range(i + 1, min(i + 3, len(thread))):  # Check next 2 messages
                    if not thread[j]['is_customer'] and thread[j]['author_id'] == BRAND:
                        # Build thread context (previous messages)
                        context = []
                        for k in range(max(0, i - 4), i):
                            role = "customer" if thread[k]['is_customer'] else "agent"
                            context.append({"role": role, "text": thread[k]['text_clean']})

                        pair = {
                            'id': f"{msg['tweet_id']}_{thread[j]['tweet_id']}",
                            'customer_text': msg['text_clean'],
                            'customer_text_raw': msg['text'],
                            'brand_reply': thread[j]['text_clean'],
                            'brand_reply_raw': thread[j]['text'],
                            'thread_context': context,
                            'thread_length': len(thread),
                            'position_in_thread': i,
                            'customer_author': msg['author_id'],
                        }
                        pairs.append(pair)
                        break  # Only take the first brand reply

    return pairs


def detect_escalation_signals(brand_reply: str) -> dict:
    """
    Detect escalation signals from the brand's actual reply.
    This gives us ground truth for escalation labels.
    """
    reply_lower = brand_reply.lower()
    signals = {
        'suggests_dm': bool(re.search(r'dm|direct message|private message|send us a message', reply_lower)),
        'suggests_phone': bool(re.search(r'call us|phone|1-800|1-888|give us a call', reply_lower)),
        'provides_link': bool(re.search(r'https?://', reply_lower)),
        'apologizes': bool(re.search(r'sorry|apologize|apologies', reply_lower)),
        'asks_for_details': bool(re.search(r'can you (share|provide|send)|please (share|provide|send)', reply_lower)),
        'indicates_resolution': bool(re.search(r"glad to hear|you're welcome|happy to help|glad we could|resolved", reply_lower)),
    }
    # Escalation = brand redirected to private channel or provided complex resolution path
    signals['is_escalation'] = signals['suggests_dm'] or signals['suggests_phone'] or (
        signals['provides_link'] and signals['apologizes']
    )
    return signals


def main():
    print("=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)

    # Step 1: Load raw data
    print("\n[1/6] Loading raw CSV...")
    df = pd.read_csv(
        str(RAW_CSV),
        dtype={
            'tweet_id': str, 'author_id': str, 'inbound': str,
            'created_at': str, 'text': str, 'response_tweet_id': str,
            'in_response_to_tweet_id': str
        }
    )
    print(f"  Total rows: {len(df):,}")

    # Step 2: Filter to AmazonHelp conversations
    print(f"\n[2/6] Filtering to {BRAND} conversations...")
    # Get all brand replies
    brand_replies = df[(df['author_id'] == BRAND) & (df['inbound'].str.lower() == 'false')]
    # Get all tweet IDs in brand conversations (customer msgs that brand replied to)
    reply_to_ids = brand_replies['in_response_to_tweet_id'].dropna().unique()
    # Get customer messages
    customer_msgs = df[df['tweet_id'].isin(reply_to_ids)]
    # Combine: all messages in conversations involving this brand
    all_brand_tweet_ids = set(brand_replies['tweet_id'].values) | set(customer_msgs['tweet_id'].values)

    # Also include follow-up messages
    response_ids = brand_replies['response_tweet_id'].dropna()
    for resp_str in response_ids:
        for rid in str(resp_str).split(','):
            rid = rid.strip()
            if rid and rid != 'nan':
                all_brand_tweet_ids.add(rid)

    brand_df = df[df['tweet_id'].isin(all_brand_tweet_ids)].copy()
    print(f"  Brand conversation messages: {len(brand_df):,}")

    # Step 3: Filter to English
    print("\n[3/6] Filtering to English messages...")
    brand_df['is_english'] = brand_df['text'].apply(is_english)
    english_df = brand_df[brand_df['is_english']].copy()
    print(f"  English messages: {len(english_df):,} ({len(english_df)/len(brand_df)*100:.1f}%)")

    # Step 4: Reconstruct threads
    print("\n[4/6] Reconstructing conversation threads...")
    threads = reconstruct_threads(english_df)
    print(f"  Threads reconstructed: {len(threads):,}")
    thread_lengths = [len(t) for t in threads]
    print(f"  Thread length: mean={np.mean(thread_lengths):.1f}, median={np.median(thread_lengths):.0f}, max={max(thread_lengths)}")

    # Step 5: Extract conversation pairs
    print("\n[5/6] Extracting conversation pairs...")
    pairs = extract_conversation_pairs(threads)
    print(f"  Total (customer, brand_reply) pairs: {len(pairs):,}")

    # Add escalation signals
    for pair in pairs:
        pair['escalation_signals'] = detect_escalation_signals(pair['brand_reply'])

    # Subsample if needed
    if len(pairs) > MAX_CONVERSATIONS:
        print(f"\n  Subsampling to {MAX_CONVERSATIONS} pairs...")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(pairs), MAX_CONVERSATIONS, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    # Step 6: Save
    print(f"\n[6/6] Saving to {PROCESSED_DATA}...")
    with open(PROCESSED_DATA, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    # Print stats
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total pairs saved: {len(pairs):,}")

    escalation_count = sum(1 for p in pairs if p['escalation_signals']['is_escalation'])
    print(f"  Escalation signals: {escalation_count:,} ({escalation_count/len(pairs)*100:.1f}%)")

    has_context = sum(1 for p in pairs if len(p['thread_context']) > 0)
    print(f"  Pairs with thread context: {has_context:,} ({has_context/len(pairs)*100:.1f}%)")

    avg_customer_len = np.mean([len(p['customer_text']) for p in pairs])
    avg_reply_len = np.mean([len(p['brand_reply']) for p in pairs])
    print(f"  Avg customer text length: {avg_customer_len:.0f} chars")
    print(f"  Avg brand reply length: {avg_reply_len:.0f} chars")

    # Sample output
    print("\n--- Sample pairs ---")
    for p in pairs[:3]:
        print(f"\n  Customer: {p['customer_text'][:150]}")
        print(f"  Reply:    {p['brand_reply'][:150]}")
        print(f"  Escalation: {p['escalation_signals']['is_escalation']}")
        print(f"  Context msgs: {len(p['thread_context'])}")


if __name__ == "__main__":
    main()
