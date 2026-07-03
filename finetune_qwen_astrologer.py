"""
Fine-tune Qwen2.5-7B-Instruct (or Qwen3-8B) on the Vedaz astrologer chat data
using LoRA with Unsloth.

WHY UNSLOTH: it's the easiest way to fine-tune Qwen on a single GPU (even a
16GB one) without running out of memory. It uses LoRA (only trains a small
set of extra weights, not the full model) so training is fast and cheap.

RUN THIS ON YOUR VPS (with GPU), NOT ON A LAPTOP WITHOUT A GPU.

Steps before running:
  1. pip install unsloth
  2. Place clean_astrologer_data.jsonl in the same folder as this script
  3. python finetune_qwen_astrologer.py
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Load base model
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # 4-bit = fits on smaller GPUs
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,          # auto-detect (bf16/fp16 depending on GPU)
    load_in_4bit=True,   # saves a lot of VRAM
)

# ---------------------------------------------------------------------------
# 2. Add LoRA adapters (this is what actually gets trained)
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                 # LoRA rank - 16 is a good default for small datasets
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---------------------------------------------------------------------------
# 3. Load and format the dataset
# ---------------------------------------------------------------------------
# clean_astrologer_data.jsonl already has the shape:
# {"messages": [{"role": "system", "content": ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
# We just need to convert each conversation into the model's chat template.

dataset = load_dataset("json", data_files="clean_astrologer_data.jsonl", split="train")


def format_conversation(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


dataset = dataset.map(format_conversation)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,          # small dataset (55 examples) -> a few epochs is enough
        learning_rate=2e-4,
        fp16=not model.config.torch_dtype == "bfloat16",
        bf16=model.config.torch_dtype == "bfloat16",
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

trainer.train()

# ---------------------------------------------------------------------------
# 5. Save the fine-tuned model
# ---------------------------------------------------------------------------
# Option A: save LoRA adapters only (small, ~50-200MB)
model.save_pretrained("qwen_astrologer_lora")
tokenizer.save_pretrained("qwen_astrologer_lora")

# Option B: merge LoRA into the base model and save full weights
# (useful if you want to serve it directly with vLLM without a LoRA loader)
model.save_pretrained_merged(
    "qwen_astrologer_merged",
    tokenizer,
    save_method="merged_16bit",
)

print("Done. Fine-tuned model saved to ./qwen_astrologer_merged")
print("You can now host ./qwen_astrologer_merged on your VPS using vLLM.")
