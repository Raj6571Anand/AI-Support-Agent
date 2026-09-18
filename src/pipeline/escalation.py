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

    # ── Negative sentiment word list ──────────────────────────────────
    NEGATIVE_WORDS = [
        'angry', 'furious', 'frustrated', 'disappointed', 'terrible', 'horrible',
        'awful', 'worst', 'hate', 'useless', 'pathetic', 'ridiculous', 'absurd',
        'unacceptable', 'disgusting', 'outrageous', 'incompetent', 'scam',
        'never again', 'waste', 'garbage', 'trash', 'lied', 'lying', 'cheat',
        'cheating', 'rip off', 'ripoff', 'stolen', 'stealing', 'ruined',
        'broken', 'destroyed', 'failed', 'failing', 'impossible', 'nightmare',
        'disaster', 'appalling', 'infuriating', 'fed up', 'sick of', 'tired of',
        'done with', 'give up', 'lost faith', 'no help', 'unhelpful', 'ignored',
        'neglected', 'abandoned', 'shameful', 'disgraceful', 'embarrassing',
    ]

    def _score_low_confidence(self, confidence):
        """Score based on classifier confidence. High score = low confidence."""
        if confidence <= 0:
            return 1.0
        if confidence >= 0.15:
            return 0.0
        return max(0.0, 1.0 - confidence / 0.15)

    def _score_account_security(self, text_lower):
        """Score based on account security keyword matches."""
        keywords = ESCALATION_KEYWORDS['account_security']
        matches = sum(1 for kw in keywords if kw in text_lower)
        return min(1.0, matches / max(1, len(keywords)) * 3)  # Amplify — even 1 match is strong

    def _score_legal(self, text_lower):
        """Score based on legal keyword matches."""
        keywords = ESCALATION_KEYWORDS['legal']
        matches = sum(1 for kw in keywords if kw in text_lower)
        return min(1.0, matches / max(1, len(keywords)) * 3)

    def _score_frustration(self, text_lower, intent):
        """Score based on frustration keywords, weighted by intent sensitivity."""
        keywords = ESCALATION_KEYWORDS['high_frustration']
        matches = sum(1 for kw in keywords if kw in text_lower)
        base_score = min(1.0, matches / max(1, len(keywords)) * 2)
        sensitive_intents = ('refund_return', 'order_issue', 'product_issue', 'account_access')
        weight = 1.0 if intent in sensitive_intents else 0.5
        return min(1.0, base_score * weight)

    def _score_long_thread(self, thread_context):
        """Score based on conversation length. Long threads = likely unresolved."""
        length = len(thread_context) if thread_context else 0
        if length <= 1:
            return 0.0
        return min(1.0, length / 6.0)

    def _score_account_access(self, intent):
        """Score 1.0 if the intent is account_access."""
        return 1.0 if intent == 'account_access' else 0.0

    def _score_sentiment(self, text_lower):
        """Score based on negative sentiment word density."""
        words = text_lower.split()
        if not words:
            return 0.0
        matches = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
        # Normalize: 1 match = 0.3, 2 = 0.5, 3+ = 0.7+
        return min(1.0, matches * 0.25)

    def _score_sensitive_intent(self, intent):
        """Score based on whether the intent typically needs human attention."""
        sensitive = {
            'account_access': 0.8,
            'refund_return': 0.4,
            'order_issue': 0.3,
            'product_issue': 0.3,
            'payment_billing': 0.3,
            'subscription': 0.2,
            'general_inquiry': 0.0,
            'feedback': 0.0,
        }
        return sensitive.get(intent, 0.1)

    def decide_scored(self, customer_text, intent, confidence, thread_context=None,
                      threshold=0.45, aggregation='max'):
        """
        Scored escalation: each criterion produces a 0-1 score.
        The aggregate score is thresholded for the final decision.
        
        Args:
            threshold: Decision boundary. Score >= threshold → escalate.
            aggregation: 'max' (any single criterion triggers) or 'weighted_mean'
        
        Returns:
            dict with decision, escalation_score, criterion_scores, threshold
        """
        if thread_context is None:
            thread_context = []

        text_lower = customer_text.lower()

        criterion_scores = {
            'low_confidence': self._score_low_confidence(confidence),
            'account_security': self._score_account_security(text_lower),
            'legal': self._score_legal(text_lower),
            'frustration': self._score_frustration(text_lower, intent),
            'long_thread': self._score_long_thread(thread_context),
            'account_access': self._score_account_access(intent),
            'sentiment': self._score_sentiment(text_lower),
            'sensitive_intent': self._score_sensitive_intent(intent),
        }

        if aggregation == 'max':
            escalation_score = max(criterion_scores.values())
        elif aggregation == 'weighted_mean':
            weights = {
                'low_confidence': 1.5,
                'account_security': 2.0,
                'legal': 2.0,
                'frustration': 1.5,
                'long_thread': 1.0,
                'account_access': 1.5,
                'sentiment': 1.0,
                'sensitive_intent': 0.5,
            }
            total_weight = sum(weights.values())
            escalation_score = sum(
                criterion_scores[k] * weights[k] for k in criterion_scores
            ) / total_weight
        else:
            escalation_score = max(criterion_scores.values())

        decision = 'escalate' if escalation_score >= threshold else 'auto_handle'

        # Build triggered list (criteria above half the threshold)
        triggered = [k for k, v in criterion_scores.items() if v >= threshold * 0.5]
        top_criterion = max(criterion_scores, key=criterion_scores.get)

        if decision == 'escalate':
            reason = f"Escalation score {escalation_score:.2f} >= {threshold:.2f} (top: {top_criterion}={criterion_scores[top_criterion]:.2f})"
        else:
            reason = f"Escalation score {escalation_score:.2f} < {threshold:.2f} — auto-handling"

        return {
            'decision': decision,
            'reason': reason,
            'criteria_triggered': triggered,
            'confidence': confidence,
            'escalation_score': escalation_score,
            'criterion_scores': criterion_scores,
            'threshold': threshold,
            'aggregation': aggregation,
            'top_criterion': top_criterion,
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
