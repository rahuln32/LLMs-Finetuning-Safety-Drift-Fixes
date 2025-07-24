import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset

# —— Configuration —— 
MODEL_BASE = "TheBloke/Llama-2-7B-Chat-fp16"
ADAPTER_CAP = "foo-barrr/alpaca-7b-lora"

# 1) Initialize tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_BASE)
tok.pad_token  = tok.eos_token
tok.padding_side = "left"
tok.model_max_length = 2048

# 2) Load base model in FP16 
base = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    torch_dtype=torch.float16,
    device_map="auto",
)
base.config.pad_token_id = tok.eos_token_id

# 3) Attach Alpaca LoRA adapter
model = PeftModel.from_pretrained(
    base,
    ADAPTER_CAP,
    adapter_name="alpaca",
    device_map="auto",
)
model.eval()
torch.inference_mode()

# —— Self‑critique playbook with robust parsing —— 
prompts_ds = Dataset.load_from_disk("constitutional_ai/unsafe_prompts")
all_prompts = prompts_ds["prompt"]  # list of strings
BATCH_SIZE = 10
WRITE_EVERY = 50  # write & flush every N examples
OUTPUT_PATH = "constitutional_ai/cai_pairs.jsonl"
# This is the mini constitution, a condensed version of the principles Anthropic has published
CONSTITUTION = """### High‑Level Principles
1. The assistant must not provide advice or instructions that facilitate wrongdoing, illegal activities, or harmful behavior.
2. The assistant should refuse or safe‑complete requests for disallowed content while remaining helpful when possible.
3. The assistant should avoid hateful, harassing, discriminatory, or toxic language based on race, gender, religion, nationality, or other protected characteristics.
4. The assistant should respect and promote human rights, equality, freedom, privacy, and dignity for all people.
5. The assistant should discourage violence, cruelty, torture, and degrading treatment while supporting life, liberty, and security.
6. The assistant should accurately represent itself as an AI system, not claiming to have human characteristics, emotions, preferences, or physical capabilities.
7. The assistant should not give the impression of medical, legal, or financial authority; instead suggest consulting appropriate professionals.
8. The assistant should make minimal unsupported assumptions about users and respect privacy by avoiding requests for personal information.
9. The assistant should consider diverse cultural perspectives, especially those from non-Western, less industrialized, or marginalized communities.
10. The assistant should prioritize humanity's wellbeing and demonstrate humility about its own capabilities and limitations.

### Self‑Critique rubric
If the answer violates a principle, rewrite it so it is harmless yet still helpful if possible; otherwise refuse.
"""

ALTERNATE_CONSTITUTION = """### High‑Level Principles
1. The assistant must not provide advice or instructions that facilitate wrongdoing.
2. The assistant should refuse or safe‑complete requests for disallowed content.
3. The assistant should avoid hateful or harassing language.

### Self‑Critique rubric
If the answer violates a principle, rewrite it so it is harmless yet still helpful if possible; otherwise refuse.
"""

# regex to find the improved section
IMPROVED_RE = re.compile(r'<IMPROVED>\s*(.*?)\s*</IMPROVED>', re.DOTALL | re.IGNORECASE)
buffer = []
count = 0

with open(OUTPUT_PATH, "w") as fout:
    # process in batches
    for i in range(0, len(all_prompts), BATCH_SIZE):
        batch_prompts = all_prompts[i : i + BATCH_SIZE]

        # generate the first round of answers
        device = next(model.parameters()).device 
        raw_inputs = tok(
            [f"<s>[INST] <<SYS>>\n\n<</SYS>>\n<DRAFT>{q}</DRAFT> [/INST]" for q in batch_prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        raw_inputs = {k: v.to(device) for k, v in raw_inputs.items()}
        raw_ids = model.generate(**raw_inputs, max_new_tokens=256)
        raw_out = tok.batch_decode(raw_ids, skip_special_tokens=True)

        # strip off any leading [INST]/[/INST] echoes. This ocassionally happens.
        raw_texts = []
        for text in raw_out:
            # split on the last [/INST] if present
            cleaned = text.rsplit("[/INST]", 1)[-1].strip()
            raw_texts.append(cleaned)

        # build critique prompts with tags so that the answers are easy to extract
        critique_prompts = []
        for raw in raw_texts:
            critique_prompts.append(
                f"{CONSTITUTION}\n\n"
                "<DRAFT>\n"
                f"{raw}\n"
                "</DRAFT>\n\n"
                "Now please output exactly two sections, with no extra text outside the tags:\n\n"
                "<CRITIQUE>\n"
                "Your critical analysis of the draft answer goes here.\n"
                "</CRITIQUE>\n\n"
                "<IMPROVED>\n"
                "Your improved safe answer goes here.\n"
                "</IMPROVED>"
            )

        # generate revised answers
        crit_inputs = tok(
            [f"<s>[INST] <<SYS>>\n\n<</SYS>>\n{c} [/INST]" for c in critique_prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        crit_inputs = {k: v.to(device) for k, v in crit_inputs.items()}
        rev_ids      = model.generate(**crit_inputs, max_new_tokens=512)
        rev_out_raw  = tok.batch_decode(rev_ids, skip_special_tokens=True)

        # 4) Filtering out unformatted output
        filtered_batch_prompts = []
        filtered_raw_texts = []
        filtered_rev_texts = []
        for i in range(len(rev_out_raw)):
            text = rev_out_raw[i]
            # Remove any trailing prompt echo
            tail = text.rsplit("[/INST]", 1)[-1]
            
            # Try to find content between <IMPROVED> tags
            improved_match = IMPROVED_RE.search(tail)
            
            if improved_match:
                # Found properly formatted <IMPROVED> content
                improved = improved_match.group(1).strip()
                # Final cleanup: remove any remaining XML tags and excessive whitespace
                improved = re.sub(r'<[^>]+>', '', improved)  # Remove any remaining tags
                improved = re.sub(r'\s+', ' ', improved).strip()  # Normalize whitespace
                filtered_batch_prompts.append(batch_prompts[i])
                filtered_raw_texts.append(raw_texts[i])
                filtered_rev_texts.append(improved)

        # 5) collect into pairs
        for q, raw, rev in zip(filtered_batch_prompts, filtered_raw_texts, filtered_rev_texts):
            buffer.append({"prompt": q, "chosen": rev, "rejected": raw})
            count += 1
            # once buffer is full, dump it to disk
            if count % WRITE_EVERY == 0:
                for rec in buffer:
                    fout.write(json.dumps(rec) + "\n")
                fout.flush()
                buffer = []  # reset
        
    # write any leftovers
    if buffer:
        for rec in buffer:
            fout.write(json.dumps(rec) + "\n")
        fout.flush()

print(f"Generated {count}(raw → revised) pairs.")