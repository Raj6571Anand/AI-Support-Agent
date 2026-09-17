"""
Data Exploration Script — Step 1
Explore the Twitter Customer Support dataset, pick the best brand,
and understand the conversation structure.
"""
import pandas as pd
import numpy as np
from collections import Counter

print("=" * 70)
print("STEP 1: Loading dataset...")
print("=" * 70)

# Load with proper dtypes to handle the messy data
df = pd.read_csv(
    r"d:\Hiver\data\twcs\twcs.csv",
    dtype={
        'tweet_id': str,
        'author_id': str,
        'inbound': str,
        'created_at': str,
        'text': str,
        'response_tweet_id': str,
        'in_response_to_tweet_id': str
    }
)

print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample row:")
print(df.iloc[0].to_dict())

print("\n" + "=" * 70)
print("STEP 2: Understanding inbound vs outbound")
print("=" * 70)

print(f"\nInbound value counts:")
print(df['inbound'].value_counts())
print(f"\nNull inbound: {df['inbound'].isna().sum()}")

# Inbound = True means customer message, False = brand reply
df['is_customer'] = df['inbound'].astype(str).str.lower() == 'true'
print(f"\nCustomer messages: {df['is_customer'].sum():,}")
print(f"Brand replies: {(~df['is_customer']).sum():,}")

print("\n" + "=" * 70)
print("STEP 3: Brand analysis — finding the best brand")
print("=" * 70)

# Brand messages = outbound (is_customer = False)
brand_msgs = df[~df['is_customer']]
brand_counts = brand_msgs['author_id'].value_counts()

print(f"\nTotal unique brands: {len(brand_counts)}")
print(f"\nTop 20 brands by reply count:")
for brand, count in brand_counts.head(20).items():
    # Count how many unique customers this brand talked to
    brand_df = df[df['author_id'] == brand]
    # Count customer messages TO this brand
    brand_reply_ids = brand_df['in_response_to_tweet_id'].dropna().unique()
    customer_msgs = df[df['tweet_id'].isin(brand_reply_ids)]
    n_customers = customer_msgs['author_id'].nunique() if len(customer_msgs) > 0 else 0
    print(f"  {brand:30s} — {count:>6,} replies, ~{n_customers:>5,} customers")

print("\n" + "=" * 70)
print("STEP 4: Thread reconstruction stats")
print("=" * 70)

# How many messages have in_response_to_tweet_id?
has_response_to = df['in_response_to_tweet_id'].notna().sum()
has_response_ids = df['response_tweet_id'].notna().sum()
print(f"Messages with in_response_to_tweet_id: {has_response_to:,}")
print(f"Messages with response_tweet_id: {has_response_ids:,}")

print("\n" + "=" * 70)
print("STEP 5: Deep dive on top 5 brands")
print("=" * 70)

for brand in brand_counts.head(5).index:
    print(f"\n--- {brand} ---")
    brand_replies = df[(df['author_id'] == brand) & (~df['is_customer'])]
    
    # Get customer messages that this brand replied to
    reply_to_ids = brand_replies['in_response_to_tweet_id'].dropna().unique()
    customer_msgs = df[df['tweet_id'].isin(reply_to_ids)]
    
    print(f"  Brand replies: {len(brand_replies):,}")
    print(f"  Customer messages replied to: {len(customer_msgs):,}")
    
    # Average text length
    avg_reply_len = brand_replies['text'].str.len().mean()
    avg_customer_len = customer_msgs['text'].str.len().mean() if len(customer_msgs) > 0 else 0
    print(f"  Avg brand reply length: {avg_reply_len:.0f} chars")
    print(f"  Avg customer msg length: {avg_customer_len:.0f} chars")
    
    # Sample customer messages
    if len(customer_msgs) > 0:
        print(f"  Sample customer messages:")
        for _, row in customer_msgs.sample(min(3, len(customer_msgs)), random_state=42).iterrows():
            text = str(row['text'])[:120]
            print(f"    → {text}")

print("\n" + "=" * 70)
print("STEP 6: Sample conversation thread")
print("=" * 70)

# Pick the top brand and reconstruct a sample thread
top_brand = brand_counts.index[0]
print(f"\nSample thread from {top_brand}:")

# Find a brand reply with in_response_to_tweet_id
sample_reply = df[(df['author_id'] == top_brand) & df['in_response_to_tweet_id'].notna()].iloc[0]
thread = [sample_reply]

# Walk back to find the customer message
current = sample_reply
for _ in range(10):  # max depth
    parent_id = current['in_response_to_tweet_id']
    parent = df[df['tweet_id'] == str(parent_id)]
    if len(parent) == 0:
        break
    current = parent.iloc[0]
    thread.insert(0, current)

# Walk forward to find follow-ups
current = sample_reply
for _ in range(10):
    response_ids = str(current.get('response_tweet_id', '')).split(',')
    response_ids = [r.strip() for r in response_ids if r.strip() and r.strip() != 'nan']
    if not response_ids:
        break
    child = df[df['tweet_id'].isin(response_ids)]
    if len(child) == 0:
        break
    current = child.iloc[0]
    thread.append(current)

for msg in thread:
    role = "CUSTOMER" if msg['is_customer'] else f"BRAND ({msg['author_id']})"
    text = str(msg['text'])[:150]
    print(f"  [{role}] {text}")

print("\n✅ Exploration complete!")
