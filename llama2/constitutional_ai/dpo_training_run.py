from peft import LoraConfig, PeftModel
import datasets
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import torch

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
    torch_dtype=torch.bfloat16, # This is important! Else, numerical instability!!!
    device_map="auto",
)
base.config.pad_token_id = tok.eos_token_id

# 3) Attach Alpaca LoRA adapter
model = PeftModel.from_pretrained(
    base,
    ADAPTER_CAP,
    adapter_name="alpaca",
    device_map="auto",
    torch_dtype=torch.bfloat16, # This is important! Else, numerical instability!!!
)

model.set_adapter("alpaca")
print("Freezing existing adapters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        param.requires_grad = False

print("Adding safety adapter:")
safety_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter("safety", safety_cfg)
model.set_adapter("safety")

model.train()  # Make sure model is in training mode

# Explicitly enable gradients for the safety adapter
for name, param in model.named_parameters():
    if "safety" in name:
        param.requires_grad = True
print("Printing trainable params. Should only show weights from the new safety adapter.")
model.print_trainable_parameters()

# Load and split dataset
ds = datasets.load_dataset("json", data_files="constitutional_ai/cai_pairs.jsonl", split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)  # 90/10 split

# Set max lengths for tokenization
tok.model_max_length = 1024
tok.truncation_side = "left"  # Truncate from left to keep response intact

# Create explicit reference model to avoid auto-cloning issues
# Load a separate instance of the base + alpaca adapter
ref_base = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    torch_dtype=torch.float16,
    device_map="auto",
)
ref_base.config.pad_token_id = tok.eos_token_id
# Load the alpaca adapter on the reference model
ref_model = PeftModel.from_pretrained(
    ref_base,
    ADAPTER_CAP,
    adapter_name="alpaca",
    device_map="auto",
)
# Freeze the reference model completely
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

print(f"   Reference model created with {sum(1 for p in ref_model.parameters())} parameters")
print(f"   Reference model trainable params: {sum(1 for p in ref_model.parameters() if p.requires_grad)}")

ref_model.set_adapter("alpaca")
ref_model.eval()

# Create working trainer with SGD
optimal_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args = DPOConfig(
        ##################################################################
        # I/O
        ##################################################################
        output_dir                 = "outputs/safety_adapter_final",
        logging_steps              = 10,
        save_strategy              = "steps",
        eval_strategy              = "steps",
        save_steps                 = 50,
        eval_steps                 = 50,
        save_total_limit           = 3,
        load_best_model_at_end     = True,
        metric_for_best_model      = "eval_loss",

        ##################################################################
        # Data & schedule
        ##################################################################
        per_device_train_batch_size = 4,      # 4 * 2 = 8 pairs / step
        gradient_accumulation_steps = 2,
        num_train_epochs            = 3,      # ≈3 passes over 900 pairs
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.10,   # 10 % of total steps

        ##################################################################
        # Optimiser & precision
        ##################################################################
        learning_rate   = 1e-5,               # start‑LR; cosine → 0
        optim           = "adamw_torch",      # full‑precision AdamW
        weight_decay    = 0.01,               # L2 on LoRA params
        max_grad_norm   = 1.0,                # clip
        bf16            = True,               # model & grads in BF16
        fp16            = False,              # (no AMP scaler needed)
        #loss_dtype      = "float32",          # up‑cast logits for exp()

        ##################################################################
        # Sequence lengths
        ##################################################################
        max_prompt_length   = 128,
        max_length          = 512,            # prompt + completion

        ##################################################################
        # DPO‑specific knobs
        ##################################################################
        beta = 1.0,                           # sharper preference signal
    ),
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
    processing_class=tok,
)

print("Starting DPO training with SGD...")
optimal_trainer.train()

print("Training completed successfully!")
optimal_trainer.save_model("outputs/safety_adapter")
print("Model saved to outputs/safety_adapter")

from peft.tuners.lora import LoraLayer
import copy

def fuse_lora_adapters(model, keep="alpaca", add="safety"):
    """
    In‑place fusion:
      • If both adapters exist in a given LoraLayer → sum weights.
      • If only `add` exists           → clone it into `keep`.
      • If `add` missing               → do nothing.
      • After pass, delete `add` adapter and make `keep` active.
    """
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue

        has_keep = keep in module.lora_A
        has_add  =  add in module.lora_A
        if not has_add:
            continue                     # nothing to merge in this layer

        if has_keep:
            # --- sum matrices ---
            module.lora_A[keep].weight.data += module.lora_A[add].weight.data
            module.lora_B[keep].weight.data += module.lora_B[add].weight.data
        else:
            # --- clone `add` -> `keep` ---
            module.lora_A[keep] = copy.deepcopy(module.lora_A[add])
            module.lora_B[keep] = copy.deepcopy(module.lora_B[add])

    # remove adapter metadata & activate survivor
    model.delete_adapter(add)
    model.set_adapter(keep)


#####################################################################
# 1) both adapters are already loaded in `model`                     #
#####################################################################
fuse_lora_adapters(model, keep="alpaca", add="safety")

#####################################################################
# 2) save the single merged adapter                                 #
#####################################################################
save_dir = "adapters/alpaca_safety_merged"
model.save_pretrained(save_dir, safe_serialization=True)
tok.save_pretrained(save_dir)         # tokenizer for convenience
print(f"Merged adapter written to {save_dir}")

for m in model.modules():
    if isinstance(m, LoraLayer):
        assert "safety" not in m.lora_A       # safety truly gone
print("Fusion sanity‑check passed.")