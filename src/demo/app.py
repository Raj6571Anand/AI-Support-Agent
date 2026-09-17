"""
Gradio Demo — Amazon Help AI Support Agent
Interactive demo where you paste a customer tweet and see:
- Intent classification + confidence
- Draft reply
- Escalation decision + reason
"""
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *

import gradio as gr
import json

# Lazy-load models to avoid slow startup on import
_classifier = None
_generator = None
_escalation = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from src.pipeline.classifier import HybridClassifier
        _classifier = HybridClassifier()
    return _classifier


def get_generator():
    global _generator
    if _generator is None:
        from src.pipeline.reply_generator import ReplyGenerator
        _generator = ReplyGenerator()
    return _generator


def get_escalation():
    global _escalation
    if _escalation is None:
        from src.pipeline.escalation import EscalationDecider
        _escalation = EscalationDecider()
    return _escalation


def process_message(customer_text, thread_context_text=""):
    """Main pipeline: classify → generate reply → decide escalation."""
    if not customer_text.strip():
        return "Please enter a customer message.", "", "", "", ""

    try:
        # Parse thread context
        thread_context = []
        if thread_context_text.strip():
            for line in thread_context_text.strip().split('\n'):
                line = line.strip()
                if line.startswith('[customer]'):
                    thread_context.append({"role": "customer", "text": line[10:].strip()})
                elif line.startswith('[agent]'):
                    thread_context.append({"role": "agent", "text": line[7:].strip()})
                else:
                    thread_context.append({"role": "customer", "text": line})

        # 1. Classify
        classifier = get_classifier()
        cls_result = classifier.classify(customer_text)
        intent = cls_result['intent']
        confidence = cls_result['confidence']
        method = cls_result['method']

        # Format intent output
        intent_display = f"**{intent.replace('_', ' ').title()}**\n\n"
        intent_display += f"- Confidence: {confidence:.3f}\n"
        intent_display += f"- Method: {method}\n"
        intent_display += f"- Description: {INTENTS[intent]['description']}"

        # Confidence bar
        conf_bar = f"{confidence:.1%} confidence"

        # 2. Generate reply
        generator = get_generator()
        reply_result = generator.generate_reply(
            customer_text, intent,
            thread_context=json.dumps(thread_context) if thread_context else None
        )
        draft_reply = reply_result['reply']

        # Retrieved examples
        retrieved = reply_result.get('retrieved_examples', [])
        retrieved_display = ""
        for i, ex in enumerate(retrieved[:3], 1):
            retrieved_display += f"**Example {i}:**\n"
            retrieved_display += f"- Customer: {ex['customer_text'][:150]}\n"
            retrieved_display += f"- Reply: {ex['brand_reply'][:150]}\n\n"

        # 3. Escalation decision
        escalation = get_escalation()
        esc_result = escalation.decide(
            customer_text, intent, confidence,
            thread_context=thread_context
        )
        esc_decision = esc_result['decision']
        esc_reason = esc_result['reason']
        esc_criteria = esc_result.get('criteria_triggered', [])

        if esc_decision == 'escalate':
            esc_display = f"🔴 **ESCALATE TO HUMAN**\n\n"
        else:
            esc_display = f"🟢 **AUTO-HANDLE**\n\n"
        esc_display += f"**Reason:** {esc_reason}\n"
        if esc_criteria:
            esc_display += f"**Criteria triggered:** {', '.join(esc_criteria)}"

        return intent_display, conf_bar, draft_reply, esc_display, retrieved_display

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "", "", "", ""


# Build the Gradio interface
def create_demo():
    with gr.Blocks(
        title="Amazon Help AI Support Agent",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🤖 Amazon Help AI Support Agent
        **Paste a customer tweet** and the agent will classify the intent, draft a reply, and decide whether to escalate.
        
        Built on real AmazonHelp Twitter data (170k conversations). Uses embedding-based classification with LLM fallback,
        RAG-based reply generation, and data-driven escalation rules.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                customer_input = gr.Textbox(
                    label="Customer Message",
                    placeholder="e.g., My package hasn't arrived and it's been 5 days past the delivery date",
                    lines=3
                )
                thread_input = gr.Textbox(
                    label="Thread Context (optional)",
                    placeholder="Previous messages, one per line:\n[customer] I ordered last week\n[agent] Let me check on that",
                    lines=3
                )
                submit_btn = gr.Button("🔍 Analyze & Respond", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Quick Examples")
                examples = gr.Examples(
                    examples=[
                        ["My package was supposed to arrive yesterday but tracking hasn't updated in 3 days"],
                        ["I want a refund, this product is completely broken"],
                        ["I can't sign into my account, it says locked"],
                        ["Cancel my Prime membership, I was charged without consent"],
                        ["Thanks for the quick help, really appreciate it!"],
                        ["How do I change my delivery address?"],
                        ["My Kindle won't turn on anymore, what can I do?"],
                        ["I'm talking to my lawyer about this, this is unacceptable fraud"],
                    ],
                    inputs=customer_input,
                    label=""
                )

        with gr.Row():
            with gr.Column():
                intent_output = gr.Markdown(label="Intent Classification")
                confidence_output = gr.Textbox(label="Confidence", interactive=False)
            with gr.Column():
                escalation_output = gr.Markdown(label="Escalation Decision")

        with gr.Row():
            reply_output = gr.Textbox(
                label="📝 Draft Reply",
                lines=4,
                interactive=False
            )

        with gr.Accordion("🔎 Retrieved Similar Conversations", open=False):
            retrieved_output = gr.Markdown()

        submit_btn.click(
            fn=process_message,
            inputs=[customer_input, thread_input],
            outputs=[intent_output, confidence_output, reply_output, escalation_output, retrieved_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)
