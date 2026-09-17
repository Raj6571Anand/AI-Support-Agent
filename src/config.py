"""
Central configuration for the Amazon Help AI Support Agent.
All paths, model names, intent definitions, and thresholds in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "twcs" / "twcs.csv"
PROCESSED_DATA = DATA_DIR / "amazon_processed.json"
GOLDEN_SET_PATH = PROJECT_ROOT / "golden_set" / "golden_set.json"
GOLDEN_SET_LABELLED = PROJECT_ROOT / "golden_set" / "golden_set_labelled.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for d in [DATA_DIR, CHROMA_DIR, RESULTS_DIR, PROJECT_ROOT / "golden_set"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Brand ──────────────────────────────────────────────────────────────
BRAND = "AmazonHelp"

# ── Models ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_GENERATION = "openai/gpt-oss-20b"      # For reply generation + classification fallback
GROQ_MODEL_JUDGE = "openai/gpt-oss-20b"            # For LLM-as-judge
GROQ_MODEL_JUDGE_2 = "qwen/qwen3.6-27b"            # Second judge for inter-rater
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"        # Local embedding model

# ── Subsample ──────────────────────────────────────────────────────────
MAX_CONVERSATIONS = 5000       # Number of conversations to process
GOLDEN_SET_SIZE = 200          # Hand-labelled evaluation examples
EXAMPLES_PER_INTENT = 25       # Stratified sampling target

# ── Intent Taxonomy ────────────────────────────────────────────────────
INTENTS = {
    "order_issue": {
        "description": "Delivery delays, missing/lost packages, wrong items delivered, tracking problems, order status inquiries",
        "examples": [
            "My package hasn't arrived and it's been 5 days past the delivery date",
            "Where is my order? The tracking hasn't updated in 3 days",
            "I received the wrong item in my package",
            "My delivery was marked as delivered but I never got it",
            "When will my order ship? I ordered a week ago",
            "The package arrived damaged and items are broken",
            "I need to change my delivery address for a pending order",
            "My order is showing as delayed with no new estimated date",
            "The courier said they delivered it but nothing was at my door",
            "I ordered two items but only received one",
        ]
    },
    "refund_return": {
        "description": "Refund requests, return process questions, overcharges, money back requests, return shipping",
        "examples": [
            "I want a refund for this defective product",
            "How do I return an item I purchased last week?",
            "I was charged twice for the same order",
            "When will I receive my refund? It's been 10 days",
            "I returned the item but haven't gotten my money back",
            "The return label isn't working, how do I send this back?",
            "I was overcharged and need the difference refunded",
            "Can I get a refund if I don't have the original packaging?",
            "My refund was less than what I paid, why?",
            "I need to return something but the return window passed",
        ]
    },
    "account_access": {
        "description": "Login problems, locked accounts, password resets, verification issues, account security",
        "examples": [
            "I can't sign into my Amazon account",
            "My account is locked and I don't know why",
            "I need to reset my password but not getting the email",
            "Someone hacked my account and changed my email",
            "I keep getting asked to verify my identity",
            "My account was suspended without explanation",
            "I can't access my order history after changing my email",
            "Two-factor authentication isn't sending me codes",
            "I need to update my phone number on my account",
            "My account shows unauthorized purchases",
        ]
    },
    "subscription": {
        "description": "Prime membership, subscription billing, cancellation of Prime or subscriptions, Prime benefits",
        "examples": [
            "Why was I charged for Prime? I didn't sign up",
            "How do I cancel my Prime membership?",
            "I want to cancel my Prime subscription and get a refund",
            "My Prime free trial ended and I was charged",
            "Prime video isn't working even though I'm a member",
            "I didn't authorize this Prime renewal charge",
            "How do I share my Prime benefits with family?",
            "I cancelled Prime but was still charged this month",
            "Is Prime worth it? What benefits do I get?",
            "My student Prime discount isn't being applied",
        ]
    },
    "product_issue": {
        "description": "Defective/damaged products, quality complaints, wrong item received, warranty claims, device issues (Kindle, Alexa, Echo)",
        "examples": [
            "The product stopped working after one month",
            "My Kindle won't turn on anymore",
            "The item I received is completely different from the listing",
            "Alexa stopped responding to voice commands",
            "The quality of this product is terrible compared to the description",
            "My Echo device keeps disconnecting from WiFi",
            "The product broke on the first use",
            "Fire tablet screen is cracked right out of the box",
            "This item is clearly a counterfeit, not the real brand",
            "My device overheats and shuts down randomly",
        ]
    },
    "payment_billing": {
        "description": "Payment failures, gift cards, promotional credits, charge disputes, payment method issues",
        "examples": [
            "My gift card balance disappeared from my account",
            "Payment failed but my bank shows the charge went through",
            "I have a promotional credit but it's not applying at checkout",
            "My credit card was charged but the order shows as unpaid",
            "How do I add a new payment method?",
            "I was charged sales tax on an exempt item",
            "My gift card code isn't being accepted",
            "I see a charge from Amazon I don't recognize",
            "The discount code I have isn't working",
            "Can I split payment between gift card and credit card?",
        ]
    },
    "feedback": {
        "description": "Praise, thanks, positive or negative sentiment without a specific actionable request, general complaints about service",
        "examples": [
            "Thanks for the quick help!",
            "Your customer service is terrible",
            "I'm never shopping on Amazon again",
            "Great service, really appreciate it",
            "You guys are the worst, absolutely useless",
            "Thank you so much for resolving this",
            "I've been a loyal customer for years and this is how you treat me",
            "Amazing experience, will recommend to friends",
            "Worst customer support ever, no one can help",
            "Thanks for following up on this",
        ]
    },
    "general_inquiry": {
        "description": "How-to questions, policy questions, general questions that don't fit other categories",
        "examples": [
            "How do I change my delivery address?",
            "What is your return policy?",
            "How long does standard shipping take?",
            "Can I get same-day delivery in my area?",
            "How do I leave a review for a product?",
            "What happens if I refuse a delivery?",
            "How do I contact a third-party seller?",
            "Is there a way to track my package in real-time?",
            "Do you offer price matching?",
            "How do I report a suspicious seller?",
        ]
    }
}

INTENT_NAMES = list(INTENTS.keys())

# ── Escalation Thresholds ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.02          # Below this → uncertain → escalate/LLM fallback
ESCALATION_KEYWORDS = {
    "account_security": ["hacked", "unauthorized", "fraud", "stolen", "identity theft", "suspicious activity"],
    "legal": ["lawyer", "attorney", "sue", "legal action", "consumer protection", "bbb", "ftc", "lawsuit"],
    "high_frustration": ["worst ever", "never again", "reporting you", "scam", "demand", "unacceptable"],
}

# ── LLM Settings ───────────────────────────────────────────────────────
MAX_TOKENS_REPLY = 280              # Twitter-length replies
MAX_TOKENS_JUDGE = 500
TEMPERATURE_REPLY = 0.3
TEMPERATURE_JUDGE = 0.1
TOP_K_RETRIEVAL = 5                 # Number of similar conversations to retrieve
