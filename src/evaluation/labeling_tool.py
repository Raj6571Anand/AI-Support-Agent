import json
import random
import os
import math
from pathlib import Path
import gradio as gr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import INTENTS, INTENT_NAMES

GOLDEN_SET_PATH = Path(r"d:\Hiver\golden_set\golden_set_labelled.json")
HUMAN_LABELS_PATH = Path(r"d:\Hiver\golden_set\human_labels.json")

def get_stratified_sample():
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by intent
    by_intent = {intent: [] for intent in INTENT_NAMES}
    for item in data:
        intent = item.get("true_intent")
        if intent in by_intent:
            by_intent[intent].append(item)
    
    # Sort for deterministic behavior before shuffling
    for intent in INTENT_NAMES:
        by_intent[intent].sort(key=lambda x: x["id"])
    
    random.seed(42)
    sample = []
    total = sum(len(items) for items in by_intent.values())
    target_total = 100
    
    # Calculate exact quotas using largest remainder method
    quotas = {}
    remainders = {}
    for intent, items in by_intent.items():
        proportion = len(items) / total * target_total
        quotas[intent] = math.floor(proportion)
        remainders[intent] = proportion - math.floor(proportion)
    
    # Distribute remaining
    shortfall = target_total - sum(quotas.values())
    sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    for i in range(shortfall):
        quotas[sorted_remainders[i][0]] += 1
        
    for intent, items in by_intent.items():
        shuffled = list(items)
        random.shuffle(shuffled)
        sample.extend(shuffled[:quotas[intent]])
        
    # Shuffle final sample
    random.shuffle(sample)
    return sample

def load_human_labels():
    if HUMAN_LABELS_PATH.exists():
        with open(HUMAN_LABELS_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_human_labels(labels):
    with open(HUMAN_LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

sample_data = get_stratified_sample()
human_labels = load_human_labels()

def format_thread(thread):
    if not thread:
        return "No thread context."
    out = ""
    for msg in thread:
        role = msg.get("role", "").capitalize()
        text = msg.get("text", "")
        out += f"**{role}**: {text}\n\n"
    return out

def get_state(index):
    if index < 0 or index >= len(sample_data):
        return None
    item = sample_data[index]
    
    # Check if already labeled
    label_info = human_labels.get(item["id"], {})
    human_intent = label_info.get("human_intent", None)
    human_escalation = label_info.get("human_escalation", None)
    notes = label_info.get("labeler_notes", "")
    
    # Reveal LLM label only if human has already labeled it
    llm_info = ""
    if human_intent is not None:
        llm_intent = item.get("true_intent")
        llm_esc = "Yes" if item.get("true_escalation") else "No"
        llm_info = f"**LLM Intent**: {llm_intent} | **LLM Escalation**: {llm_esc}"
        
    return {
        "id": item["id"],
        "customer_text": item.get("customer_text", ""),
        "thread_context": format_thread(item.get("thread_context", [])),
        "brand_reply_actual": item.get("brand_reply_actual", ""),
        "human_intent": human_intent,
        "human_escalation": "Yes" if human_escalation else "No" if human_escalation is False else None,
        "notes": notes,
        "llm_info": llm_info,
        "progress": f"{index + 1} / {len(sample_data)}"
    }

def update_ui(index):
    state = get_state(index)
    if not state:
        return [gr.update()] * 8
    
    return (
        state["progress"],
        state["customer_text"],
        state["thread_context"],
        state["brand_reply_actual"],
        state["human_intent"],
        state["human_escalation"],
        state["notes"],
        state["llm_info"]
    )

def on_submit(index, intent, escalation, notes):
    if not intent or not escalation:
        return gr.update(value="Please select both Intent and Escalation."), gr.update()
    
    item = sample_data[index]
    human_labels[item["id"]] = {
        "id": item["id"],
        "customer_text": item.get("customer_text", ""),
        "human_intent": intent,
        "human_escalation": True if escalation == "Yes" else False,
        "llm_intent": item.get("true_intent"),
        "llm_escalation": item.get("true_escalation", False),
        "labeler_notes": notes
    }
    save_human_labels(human_labels)
    
    llm_intent = item.get("true_intent")
    llm_esc = "Yes" if item.get("true_escalation") else "No"
    llm_info = f"**Saved successfully!**\n\n**LLM Intent**: {llm_intent} | **LLM Escalation**: {llm_esc}"
    return llm_info, index  # Return current index to not change page automatically

def go_next(index):
    new_index = min(len(sample_data) - 1, index + 1)
    return [new_index] + list(update_ui(new_index))

def go_prev(index):
    new_index = max(0, index - 1)
    return [new_index] + list(update_ui(new_index))

# UI Builder
with gr.Blocks(title="Hiver Golden Set Labeling Tool") as app:
    current_index = gr.State(0)
    
    with gr.Row():
        gr.Markdown("# Hiver Golden Set Labeling Tool")
        progress_text = gr.Markdown("1 / 100")
        
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Customer Context")
            customer_text = gr.Textbox(label="Customer Text", lines=3, interactive=False)
            thread_context = gr.Markdown(label="Thread Context")
            brand_reply = gr.Textbox(label="Actual Brand Reply", lines=2, interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### Intent & Descriptions")
            intent_desc_str = ""
            for k, v in INTENTS.items():
                intent_desc_str += f"- **{k}**: {v['description']}\n"
            with gr.Accordion("Show Intent Descriptions", open=False):
                gr.Markdown(intent_desc_str)
                
            intent_radio = gr.Radio(choices=INTENT_NAMES, label="Human Intent")
            escalation_radio = gr.Radio(choices=["Yes", "No"], label="Needs Escalation?")
            notes_input = gr.Textbox(label="Labeler Notes (Optional)", lines=2)
            
            submit_btn = gr.Button("Submit & Reveal LLM Label", variant="primary")
            llm_display = gr.Markdown("")
            
            with gr.Row():
                prev_btn = gr.Button("Previous")
                next_btn = gr.Button("Next")
                
    # Event wiring
    submit_btn.click(
        fn=on_submit,
        inputs=[current_index, intent_radio, escalation_radio, notes_input],
        outputs=[llm_display, current_index]
    )
    
    next_btn.click(
        fn=go_next,
        inputs=[current_index],
        outputs=[current_index, progress_text, customer_text, thread_context, brand_reply, intent_radio, escalation_radio, notes_input, llm_display]
    )
    
    prev_btn.click(
        fn=go_prev,
        inputs=[current_index],
        outputs=[current_index, progress_text, customer_text, thread_context, brand_reply, intent_radio, escalation_radio, notes_input, llm_display]
    )
    
    app.load(
        fn=update_ui,
        inputs=[current_index],
        outputs=[progress_text, customer_text, thread_context, brand_reply, intent_radio, escalation_radio, notes_input, llm_display]
    )

if __name__ == "__main__":
    app.launch()
