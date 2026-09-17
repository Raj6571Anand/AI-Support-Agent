import sys
import json
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingClassifier:
    """Classifies intents using embeddings and cosine similarity to intent centroids."""
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.centroids = {}
        self._compute_centroids()
        
    def _compute_centroids(self):
        """Computes the centroid embedding for each intent based on its examples."""
        for intent_name, data in INTENTS.items():
            examples = data.get("examples", [])
            if examples:
                embeddings = self.model.encode(examples)
                centroid = np.mean(embeddings, axis=0)
                self.centroids[intent_name] = centroid / np.linalg.norm(centroid)
            else:
                self.centroids[intent_name] = np.zeros(self.model.get_sentence_embedding_dimension())
                
    def classify(self, text: str):
        """Classifies text based on cosine similarity to centroids."""
        text_emb = self.model.encode([text])[0]
        text_emb_norm = text_emb / np.linalg.norm(text_emb)
        
        scores = {}
        for intent_name, centroid in self.centroids.items():
            if np.linalg.norm(centroid) > 0:
                scores[intent_name] = float(np.dot(text_emb_norm, centroid))
            else:
                scores[intent_name] = 0.0
                
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_intents[0]
        second_best_score = sorted_intents[1][1] if len(sorted_intents) > 1 else 0.0
        
        # confidence is the margin between the top two scores
        confidence = best_score - second_best_score
        
        return best_intent, confidence, scores


class LLMClassifier:
    """Classifies intents using a Large Language Model via Groq API."""
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL_GENERATION
        
    def classify(self, text: str):
        """Classifies text by sending a prompt to the LLM."""
        intent_descriptions = []
        for intent_name, data in INTENTS.items():
            intent_descriptions.append(f"- {intent_name}: {data.get('description', '')}")
        intent_list_str = "\n".join(intent_descriptions)
        
        prompt = (
            "You are an intent classification system for a customer support agent.\n"
            "Your task is to classify the user's message into exactly one of the following intents:\n"
            f"{intent_list_str}\n\n"
            "Output your response strictly as a JSON object with three keys: 'intent' (the intent name), "
            "'confidence' (a float between 0.0 and 1.0), and 'reasoning' (a brief explanation)."
        )
        
        user_message = f"User message: {text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            
            intent = result.get("intent", "general_inquiry")
            if intent not in INTENT_NAMES:
                intent = "general_inquiry"
                
            confidence = float(result.get("confidence", 0.0))
            return intent, confidence, result
        except Exception as e:
            logger.error(f"LLM Classification failed: {e}")
            return "general_inquiry", 0.0, {"error": str(e)}


class HybridClassifier:
    """Combines EmbeddingClassifier and LLMClassifier."""
    def __init__(self):
        self.embedding_classifier = EmbeddingClassifier()
        self.llm_classifier = LLMClassifier()
        
    def classify(self, text: str):
        """Classifies text using embeddings, falling back to LLM if confidence is low."""
        intent, confidence, scores = self.embedding_classifier.classify(text)
        
        if confidence >= CONFIDENCE_THRESHOLD:
            return {
                "intent": intent,
                "confidence": confidence,
                "method": "embedding",
                "scores": scores
            }
            
        logger.info(f"Embedding confidence ({confidence:.2f}) below threshold. Falling back to LLM.")
        llm_intent, llm_confidence, llm_result = self.llm_classifier.classify(text)
        
        return {
            "intent": llm_intent,
            "confidence": llm_confidence,
            "method": "llm",
            "scores": llm_result
        }


class KeywordClassifier:
    """Baseline classifier using simple keyword matching."""
    def __init__(self):
        self.keywords = {
            "order_issue": ['order', 'deliver', 'shipping', 'package', 'tracking', 'arrived', 'shipped'],
            "refund_return": ['refund', 'return', 'money back', 'charged twice', 'overcharged'],
            "account_access": ['account', 'password', 'login', 'sign in', 'locked', 'verify', 'hacked'],
            "subscription": ['prime', 'membership', 'subscription', 'renewal'],
            "product_issue": ['broken', 'defective', 'not working', 'damaged', 'quality', 'kindle', 'alexa', 'echo'],
            "payment_billing": ['payment', 'gift card', 'credit card', 'charge', 'transaction', 'promo'],
            "feedback": ['thank', 'thanks', 'great', 'worst', 'terrible', 'horrible', 'love', 'amazing'],
            "general_inquiry": ['how do', 'what is', 'policy', 'can i', 'where']
        }
        
    def classify(self, text: str):
        """Classifies text by counting keyword matches per intent."""
        text_lower = text.lower()
        counts = {intent: 0 for intent in self.keywords}
        
        for intent, kw_list in self.keywords.items():
            for kw in kw_list:
                if kw in text_lower:
                    counts[intent] += 1
                    
        best_intent = max(counts, key=counts.get)
        max_count = counts[best_intent]
        
        if max_count == 0:
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "method": "keyword"
            }
            
        return {
            "intent": best_intent,
            "confidence": 1.0,
            "method": "keyword"
        }

def main():
    print("Testing Keyword Classifier...")
    kc = KeywordClassifier()
    print(kc.classify("Where is my package?"))
    
    print("\nLoading Hybrid Classifier...")
    try:
        hc = HybridClassifier()
        
        test_messages = [
            "My screen arrived cracked, I need a replacement.",
            "I want to cancel my prime membership.",
            "Can you tell me your return policy?"
        ]
        
        for msg in test_messages:
            print(f"\nMessage: {msg}")
            result = hc.classify(msg)
            print(f"Result: {result}")
    except Exception as e:
        print(f"Error during Hybrid Classifier testing: {e}")

if __name__ == "__main__":
    main()
