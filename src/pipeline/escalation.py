import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

from groq import Groq

class EscalationDecider:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def decide(self, customer_text, intent, confidence, thread_context=None):
        if thread_context is None:
            thread_context = []
            
        # 1. Low classifier confidence
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                'decision': 'escalate',
                'reason': f"Classifier confidence too low ({confidence:.2f}), message intent is ambiguous",
                'criteria_triggered': ['low_confidence'],
                'confidence': confidence
            }
            
        text_lower = customer_text.lower()
        
        # 2. Account security signals
        if any(keyword in text_lower for keyword in ESCALATION_KEYWORDS['account_security']):
            return {
                'decision': 'escalate',
                'reason': "Potential account security issue detected — requires human verification",
                'criteria_triggered': ['account_security_keyword'],
                'confidence': confidence
            }
            
        # 3. Legal signals
        if any(keyword in text_lower for keyword in ESCALATION_KEYWORDS['legal']):
            return {
                'decision': 'escalate',
                'reason': "Legal language detected — requires human review",
                'criteria_triggered': ['legal_keyword'],
                'confidence': confidence
            }
            
        # 4. High frustration + sensitive intent
        sensitive_intents = ('refund_return', 'order_issue', 'product_issue')
        if intent in sensitive_intents and any(keyword in text_lower for keyword in ESCALATION_KEYWORDS['high_frustration']):
            return {
                'decision': 'escalate',
                'reason': "High customer frustration on sensitive issue — human empathy needed",
                'criteria_triggered': ['high_frustration_sensitive_intent'],
                'confidence': confidence
            }
            
        # 5. Long thread context
        if len(thread_context) > 4:
            return {
                'decision': 'escalate',
                'reason': f"Extended conversation ({len(thread_context)} messages) without resolution — likely needs human intervention",
                'criteria_triggered': ['long_thread_context'],
                'confidence': confidence
            }
            
        # 6. Account access intent
        if intent == 'account_access':
            return {
                'decision': 'escalate',
                'reason': "Account access issues require secure human verification",
                'criteria_triggered': ['account_access_intent'],
                'confidence': confidence
            }
            
        # 7. Otherwise: auto_handle
        if intent == 'feedback':
            reason = "Positive/neutral feedback — auto-response appropriate"
        elif intent == 'general_inquiry':
            reason = "Standard inquiry — can be handled with available information"
        else:
            reason = f"Clear intent ({intent}) with sufficient confidence ({confidence:.2f}) — standard resolution path"
            
        return {
            'decision': 'auto_handle',
            'reason': reason,
            'criteria_triggered': [],
            'confidence': confidence
        }

    def decide_baseline(self, customer_text, intent):
        if intent in ('feedback', 'general_inquiry'):
            return {
                'decision': 'auto_handle',
                'reason': f"Intent ({intent}) is safe for auto-handling",
                'criteria_triggered': [],
                'confidence': 1.0
            }
        else:
            return {
                'decision': 'escalate',
                'reason': f"Baseline always escalates {intent}",
                'criteria_triggered': ['baseline_escalation'],
                'confidence': 1.0
            }

def main():
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
        
    decider = EscalationDecider()
    
    test_cases = [
        {
            "text": "I can't log in",
            "intent": "account_access",
            "confidence": 0.9,
            "thread": []
        },
        {
            "text": "This is ridiculous, I want a refund now!",
            "intent": "refund_return",
            "confidence": 0.85,
            "thread": []
        },
        {
            "text": "How do I return this?",
            "intent": "refund_return",
            "confidence": 0.2,
            "thread": []
        },
        {
            "text": "I'm talking to my lawyer",
            "intent": "general_inquiry",
            "confidence": 0.9,
            "thread": []
        },
        {
            "text": "Where is my order?",
            "intent": "order_issue",
            "confidence": 0.8,
            "thread": ["Q1", "A1", "Q2", "A2", "Q3"] 
        },
        {
            "text": "How late are you open?",
            "intent": "general_inquiry",
            "confidence": 0.88,
            "thread": []
        }
    ]
    
    print("Testing EscalationDecider:")
    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: '{tc['text']}'")
        print(f"Intent: {tc['intent']}, Confidence: {tc['confidence']}")
        try:
            decision = decider.decide(tc['text'], tc['intent'], tc['confidence'], tc['thread'])
            print(f"Decision: {decision['decision']}")
            print(f"Reason: {decision['reason']}")
            print(f"Criteria: {decision['criteria_triggered']}")
        except Exception as e:
            print(f"Error evaluating test case: {e}")

if __name__ == '__main__':
    main()
