import sys
import io
from pathlib import Path

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *
from src.pipeline import llm_cache

from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

class ReplyGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(name='amazon_support')
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def retrieve_similar(self, customer_text, intent=None, top_k=TOP_K_RETRIEVAL):
        embedding = self.embedding_model.encode([f"Represent this sentence: {customer_text}"])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        
        retrieved = []
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                score = results['distances'][0][i] if results.get('distances') else 0.0
                
                customer_msg = meta.get('customer_text', doc)
                brand_msg = meta.get('brand_reply', '')
                
                retrieved.append({
                    'customer_text': customer_msg,
                    'brand_reply': brand_msg,
                    'similarity_score': score
                })
                
        return retrieved

    def generate_reply(self, customer_text, intent, thread_context=None, retrieved_examples=None):
        if retrieved_examples is None:
            retrieved_examples = self.retrieve_similar(customer_text, intent)
            
        examples_text = ""
        for ex in retrieved_examples:
            examples_text += f"Customer: {ex['customer_text']}\nAmazon Reply: {ex['brand_reply']}\n\n"
            
        context_str = thread_context if thread_context else "None"
        
        prompt = f"""You are a customer support agent for Amazon. Your tone should be professional, empathetic, and helpful — matching Amazon's real support style.

The customer's issue has been classified as: {intent}

Here are similar past conversations and how Amazon resolved them:
{examples_text}

Thread context (previous messages in this conversation):
{context_str}

Current customer message:
{customer_text}

Draft a reply that:
1. Acknowledges the customer's specific issue
2. Provides a clear resolution or next step
3. Matches Amazon's professional support tone
4. Is concise (Twitter-length, under 280 characters)

Reply directly as the Amazon support agent. Do not include any prefix like 'Amazon:' or any thinking."""

        messages = [
            {"role": "user", "content": prompt + "\n/no_think"}
        ]
        
        response = llm_cache.cached_completion(
            self.client, GROQ_MODEL_GENERATION,
            messages=messages,
            max_tokens=MAX_TOKENS_REPLY,
            temperature=TEMPERATURE_REPLY
        )
        
        reply = response.choices[0].message.content.strip()
        
        return {
            'reply': reply,
            'retrieved_examples': retrieved_examples,
            'intent': intent,
            'model_used': GROQ_MODEL_GENERATION
        }

    def generate_reply_baseline(self, customer_text, intent):
        templates = {
            "order_issue": "We're sorry about the trouble with your order. Please share your order ID via DM so we can look into this for you. ^AA",
            "refund_return": "We understand your concern about the refund. Please contact us via DM with your order details and we'll assist you right away. ^AA",
            "account_access": "We're sorry you're having trouble accessing your account. For security, please reach out via DM so we can help you securely. ^AA",
            "subscription": "We'd be happy to help with your Prime membership concern. Please send us a DM with your account details. ^AA",
            "product_issue": "We're sorry to hear about the issue with your product. Please DM us with the order ID so we can assist you. ^AA",
            "payment_billing": "We understand your concern about the charge. Please reach out via DM with your account info so we can review this. ^AA",
            "feedback": "Thank you for your feedback! We value your input and will use it to improve our service. ^AA",
            "general_inquiry": "Thanks for reaching out! Please visit our Help page at https://www.amazon.com/help for detailed information, or DM us for personalized assistance. ^AA"
        }
        
        reply = templates.get(intent, "Thanks for reaching out! Please DM us with your details so we can assist you. ^AA")
        return reply

def main():
    print("Testing Reply Generator...")
    try:
        generator = ReplyGenerator()
        
        test_messages = [
            ("Where is my order? It was supposed to arrive yesterday.", "order_issue"),
            ("I want to cancel my Prime membership, it's too expensive.", "subscription")
        ]
        
        for msg, intent in test_messages:
            print(f"\nMessage: {msg}")
            print(f"Intent: {intent}")
            print("Baseline Reply:")
            print(generator.generate_reply_baseline(msg, intent))
            
            try:
                result = generator.generate_reply(msg, intent)
                print("LLM Reply:")
                print(result['reply'])
            except Exception as e:
                print(f"Error generating LLM reply: {e}")
                
    except Exception as e:
        print(f"Error initializing: {e}")

if __name__ == "__main__":
    main()
