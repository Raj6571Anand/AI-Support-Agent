# Decision Log

Non-obvious decisions made during development, with reasoning.

## 1. Brand Selection: AmazonHelp over AppleSupport
**Why**: AmazonHelp has 170k replies (vs 107k for Apple), more diverse issue types (orders, Prime, devices, payments), and more consistent resolution patterns. Apple's issues are often too technical for a general support agent.

## 2. Subsample of 5,000 Conversations (not full 113k)
**Why**: The assignment says "a subsample is expected and encouraged." 5,000 gives statistical power for evaluation while keeping reproduction under 10 minutes. The vector store + embeddings take ~2.5 min on 5k rows.

## 3. 8 Intents (not 4, not 15)
**Why**: Too few (4) loses signal between "refund" and "billing" which need different resolution paths. Too many (15+) creates ambiguity — "order delayed" vs "order lost" vs "order tracking" all resolve the same way. 8 intents balance discriminability with actionability.

## 4. Merged Kindle/Alexa/Echo into `product_issue` Instead of Separate Intent
**Why**: The resolution path is identical — "contact manufacturer" or "check warranty." A separate "device" intent would add taxonomy complexity without changing the agent's behavior.

## 5. Merged Cancellation into `subscription` and `order_issue`
**Why**: "Cancel my Prime" and "cancel my order" have completely different resolution paths. A standalone "cancel" intent would mix these, hurting classification.

## 6. Embedding Classifier as Primary, LLM as Fallback (not LLM-first)
**Why**: Embedding classification is ~100x faster, free (no API calls), deterministic, and works offline. LLM fallback only fires for ambiguous cases (~15-20% of messages). This makes the system fast and cheap in production.

## 7. BAAI/bge-small-en-v1.5 Over all-MiniLM-L6-v2
**Why**: BGE-small-en-v1.5 scores higher on MTEB retrieval benchmarks (especially for short text) while being the same size. Both run on CPU in <3s per batch.

## 8. Confidence = Margin Between Top Two Intent Scores (not absolute score)
**Why**: Cosine similarity to centroids gives high absolute scores even for wrong classes (all >0.5). The margin between first and second best is a much better indicator of classifier certainty. Threshold of 0.35 was chosen by inspecting score distributions.

## 9. Escalation Rules Are Data-Derived, Not Assumed
**Why**: We mined AmazonHelp's actual behavior — 2.4% of replies suggest DMs, 4.0% mention phone, 41.3% contain links. The escalation criteria mirror what the brand actually does, not what we think they should do.

## 10. `account_access` Always Escalates
**Why**: In the real data, Amazon consistently redirects account issues to secure channels (DM, phone, email). Auto-handling account access could expose customer PII. This is the one intent where false-positive escalation is strictly better than false-negative.

## 11. LLM-Assisted Labeling for Golden Set (not fully manual)
**Why**: Manually labeling 200 examples from scratch would take 3-4 hours and introduce fatigue-based errors. Using LLM to suggest labels (then reviewing) is faster and more consistent. We document this honestly as "LLM-assisted" labeling.

## 12. Two LLM Judges (not one) for Reply Evaluation
**Why**: A single judge's scores are opinions, not ground truth. Two judges + Cohen's kappa quantifies how reliable the evaluation is. Where judges disagree = interesting failure cases worth examining.

## 13. Template-Based Baseline for Reply Generation (not random)
**Why**: A random baseline is meaninglessly bad. A template baseline represents "what a rule-based system could do" — which is the real alternative the business would consider. Our system needs to beat templates, not gibberish.

## 14. Thread Context Limited to Last 4 Messages
**Why**: Twitter conversations can go 20+ turns. Including all context floods the LLM prompt and dilutes recent information. 4 messages captures the current issue state without noise.

## 15. English-Only Filter (49.6% of unmatched were non-English)
**Why**: The dataset is multilingual (Japanese, Portuguese, French, German observed). Building a multilingual system is a different project. We filter to English and state this explicitly rather than silently failing on non-English input.
