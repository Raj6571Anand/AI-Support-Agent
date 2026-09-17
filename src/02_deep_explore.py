"""
Data Exploration Script — Step 2 (continued)
Deep dive on top brands + sample threads
"""
import pandas as pd
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("Loading dataset...")
df = pd.read_csv(
    r"d:\Hiver\data\twcs\twcs.csv",
    dtype={
        'tweet_id': str, 'author_id': str, 'inbound': str,
        'created_at': str, 'text': str, 'response_tweet_id': str,
        'in_response_to_tweet_id': str
    }
)
df['is_customer'] = df['inbound'].astype(str).str.lower() == 'true'

print("=" * 70)
print("DEEP DIVE: AmazonHelp")
print("=" * 70)

brand = 'AmazonHelp'
brand_replies = df[(df['author_id'] == brand) & (~df['is_customer'])]
reply_to_ids = brand_replies['in_response_to_tweet_id'].dropna().unique()
customer_msgs = df[df['tweet_id'].isin(reply_to_ids)]

print(f"Brand replies: {len(brand_replies):,}")
print(f"Customer messages replied to: {len(customer_msgs):,}")
print(f"Avg brand reply length: {brand_replies['text'].str.len().mean():.0f} chars")
print(f"Avg customer msg length: {customer_msgs['text'].str.len().mean():.0f} chars")

# Sample customer messages
print("\n--- Sample CUSTOMER messages ---")
samples = customer_msgs.sample(15, random_state=42)
for i, (_, row) in enumerate(samples.iterrows()):
    text = str(row['text'])[:200]
    print(f"  [{i+1}] {text}")

# Sample brand replies
print("\n--- Sample BRAND replies ---")
samples = brand_replies.sample(15, random_state=42)
for i, (_, row) in enumerate(samples.iterrows()):
    text = str(row['text'])[:200]
    print(f"  [{i+1}] {text}")

# ========================================
# Thread reconstruction — full conversations
# ========================================
print("\n" + "=" * 70)
print("RECONSTRUCTING SAMPLE THREADS")
print("=" * 70)

def reconstruct_thread(df, start_tweet_id, max_depth=15):
    """Walk backward + forward to reconstruct full thread."""
    thread = []
    
    # Walk backward from start
    current_id = start_tweet_id
    backward = []
    for _ in range(max_depth):
        rows = df[df['tweet_id'] == str(current_id)]
        if len(rows) == 0:
            break
        row = rows.iloc[0]
        backward.insert(0, row)
        parent_id = row['in_response_to_tweet_id']
        if pd.isna(parent_id) or str(parent_id) == 'nan':
            break
        current_id = parent_id
    
    thread = backward
    
    # Walk forward from start
    current_id = start_tweet_id
    for _ in range(max_depth):
        rows = df[df['tweet_id'] == str(current_id)]
        if len(rows) == 0:
            break
        row = rows.iloc[0]
        resp_ids = str(row.get('response_tweet_id', ''))
        if resp_ids == 'nan' or not resp_ids:
            break
        first_resp = resp_ids.split(',')[0].strip()
        child = df[df['tweet_id'] == first_resp]
        if len(child) == 0:
            break
        thread.append(child.iloc[0])
        current_id = first_resp
    
    return thread

# Find some good multi-turn threads for AmazonHelp
brand_with_responses = brand_replies[
    brand_replies['response_tweet_id'].notna() & 
    brand_replies['in_response_to_tweet_id'].notna()
]

print(f"\nBrand replies that are part of multi-turn threads: {len(brand_with_responses):,}")

# Sample 5 threads
thread_starters = brand_with_responses.sample(5, random_state=123)
for idx, (_, starter) in enumerate(thread_starters.iterrows()):
    thread = reconstruct_thread(df, starter['tweet_id'])
    if len(thread) >= 3:  # only show interesting threads
        print(f"\n--- Thread {idx+1} ({len(thread)} messages) ---")
        for msg in thread:
            role = "CUSTOMER" if msg['is_customer'] else "AMAZON"
            text = str(msg['text'])[:180]
            print(f"  [{role:8s}] {text}")

# ========================================
# Analyze escalation signals in AmazonHelp
# ========================================
print("\n" + "=" * 70)
print("ESCALATION SIGNAL ANALYSIS")
print("=" * 70)

brand_texts = brand_replies['text'].str.lower()

# DM/private message signals
dm_signals = brand_texts.str.contains('dm|direct message|private message|send us a message', na=False)
print(f"Replies suggesting DM: {dm_signals.sum():,} ({dm_signals.mean()*100:.1f}%)")

# Phone/call signals  
phone_signals = brand_texts.str.contains('call us|phone|1-800|1-888|give us a call', na=False)
print(f"Replies mentioning phone: {phone_signals.sum():,} ({phone_signals.mean()*100:.1f}%)")

# Link signals
link_signals = brand_texts.str.contains('https?://', na=False, regex=True)
print(f"Replies with links: {link_signals.sum():,} ({link_signals.mean()*100:.1f}%)")

# Apology signals (complex issues)
apology_signals = brand_texts.str.contains("sorry|apologize|apologies|i understand your frustration", na=False)
print(f"Replies with apologies: {apology_signals.sum():,} ({apology_signals.mean()*100:.1f}%)")

# Quick resolution (simple issues)
quick_signals = brand_texts.str.contains("glad to hear|you're welcome|happy to help|glad we could", na=False)
print(f"Replies indicating resolution: {quick_signals.sum():,} ({quick_signals.mean()*100:.1f}%)")

# ========================================
# Intent discovery from customer messages
# ========================================
print("\n" + "=" * 70)
print("INTENT DISCOVERY — Customer message patterns")
print("=" * 70)

cust_texts = customer_msgs['text'].str.lower()

keywords = {
    'order/delivery': ['order', 'deliver', 'shipping', 'package', 'tracking', 'arrived', 'shipped'],
    'refund/return': ['refund', 'return', 'money back', 'charged', 'billing'],
    'account': ['account', 'password', 'login', 'sign in', 'locked', 'verify'],
    'product_issue': ['broken', 'defective', 'not working', 'damaged', 'quality', 'wrong item'],
    'cancel': ['cancel', 'cancellation', 'unsubscribe'],
    'prime': ['prime', 'membership', 'subscription'],
    'customer_service': ['worst', 'terrible', 'horrible', 'disappointed', 'frustrated', 'angry'],
    'praise': ['thank', 'thanks', 'great', 'awesome', 'love', 'amazing'],
    'kindle/device': ['kindle', 'alexa', 'echo', 'fire', 'device'],
    'payment': ['payment', 'credit card', 'gift card', 'charge', 'transaction'],
}

print(f"\nKeyword-based intent distribution (from {len(customer_msgs):,} customer messages):")
for intent, kws in keywords.items():
    pattern = '|'.join(kws)
    matches = cust_texts.str.contains(pattern, na=False)
    count = matches.sum()
    pct = count / len(customer_msgs) * 100
    print(f"  {intent:25s}: {count:>6,} ({pct:5.1f}%)")

# Check overlap
no_match = ~cust_texts.str.contains('|'.join([kw for kws in keywords.values() for kw in kws]), na=False)
print(f"  {'(no keyword match)':25s}: {no_match.sum():>6,} ({no_match.mean()*100:5.1f}%)")

print("\n--- 10 random 'no keyword match' messages ---")
unmatched = customer_msgs[no_match.values].sample(min(10, no_match.sum()), random_state=42)
for i, (_, row) in enumerate(unmatched.iterrows()):
    print(f"  [{i+1}] {str(row['text'])[:200]}")

print("\nDone!")
