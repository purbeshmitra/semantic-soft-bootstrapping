import os
import re
import json
import torch
import numpy as np
import wandb
import torch.nn.functional as F

from torch import nn
from unsloth import FastLanguageModel
from datasets import Dataset as HFDataset
from pathlib import Path
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, List


major_version, minor_version = torch.cuda.get_device_capability()
use_bfloat16 = major_version >= 8
print(use_bfloat16)


chosen_model = "Qwen/Qwen2.5-3B-Instruct"
model_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

#creating a base copy for the whole training. adapters will be added later on top of it.
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = chosen_model,
    max_seq_length = 8192,   # Context length
    dtype = model_dtype,
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)

student_model = FastLanguageModel.get_peft_model(
    base_model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # rank stabilized LoRA
    loftq_config = None,  # LoftQ
)
student_model.to(model_dtype)

# Point this to wherever your logits folder is
teacher_logits_dir = Path("teacher_logits")

# Rebuild teacher_index from filenames
teacher_index = {}

# Glob all NPZ files that look like teacher logits
for path in sorted(teacher_logits_dir.glob("teacher_logits_*.npz")):
    m = re.match(r"teacher_logits_(\d+)\.npz", path.name)
    if m:
        idx = m.group(1)           # this is the string index, e.g. "0", "1", ...
        teacher_index[idx] = str(path)  # store full path as string

print(f"Reconstructed teacher_index with {len(teacher_index)} entries.")

# (Re)save index.json for future sessions
with open(teacher_logits_dir / "index.json", "w") as f:
    json.dump(teacher_index, f)
print("Saved teacher_logits/index.json")

from datasets import Dataset as HFDataset

# ---------- Prepare a student dataset that references the teacher files ----------
# k-th student example corresponds to the k-th teacher example

def attach_teacher_path_to_student(ds_student, teacher_dir):
    # ds_student is HF dataset
    # We'll create list-of-dicts with student_text and teacher_logits_path
    new_examples = []
    for i, ex in enumerate(ds_student):
        # Use index mapping 1:1; adjust mapping logic if different
        teacher_path = teacher_index.get(str(i), "")
        # flatten student messages into student prompt string
        student_messages = ex["messages"]
        # student prompt is concatenation of user messages
        # use only user contents to build prompt
        system_texts = [m["content"] for m in student_messages if m["role"] == "system"]
        user_texts = [m["content"] for m in student_messages if m["role"] == "user"]
        student_prompt_text = " ".join(system_texts + user_texts).strip()
        new_examples.append({"text": student_prompt_text, "teacher_logits_path": teacher_path})
    return HFDataset.from_list(new_examples)


def load_jsonl_to_hfdataset(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return HFDataset.from_list(data)
teacher_dataset = load_jsonl_to_hfdataset("generated_math_teacher_data.jsonl")
student_dataset = load_jsonl_to_hfdataset("generated_math_student_data.jsonl")

student_prepared = attach_teacher_path_to_student(student_dataset, teacher_logits_dir)
print("Prepared student dataset with teacher path attached; examples:", len(student_prepared))

device='cuda'

BATCH_SIZE = 4
LR = 2e-4
TEMP = 2.0
KD_WEIGHT = 1.0   # 1 for pure KD; you can mix with CE if desired
CE_WEIGHT = 1 - KD_WEIGHT
OUTPUT_DIR = "kd_trainer_output"
FP16 = True if torch.cuda.is_available() else False
PAD_TOKEN_IS_EOS = True
NUM_EPOCHS = 3
LEARNING_RATE =2e-4
MAX_SEQ_LENGTH = 8192

vocab_size = student_model.model.lm_head.out_features
print(f"Using model's actual vocab size: {vocab_size}")

# ---------- Data collator (build combined inputs and teacher_logits tensor) ----------
class DistillDataCollator:
    """
    Produces a batch for KD:
      - input_ids: [prompt_ids + reply_ids] padded
      - attention_mask
      - labels: -100 on prompt positions, reply token ids on KD positions
      - teacher_logits: (B, S, V) with zeros on non-KD positions

    We ONLY distill on the reply tokens, not on the question/prompt.
    """

    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token is None:
            if PAD_TOKEN_IS_EOS and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer requires a pad token; set PAD_TOKEN_IS_EOS or add pad token."
                )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(features)

        prompt_ids_list = []
        reply_ids_list = []
        reply_logits_list = []
        prompt_lens = []
        reply_lens = []

        # ---- Per-example processing: tokenize prompt, load teacher reply, truncate safely ----
        for f in features:
            text = f["text"]

            # Tokenize prompt WITHOUT truncation; we’ll handle max_length ourselves
            p_tok = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            p_ids = p_tok["input_ids"][0].tolist()
            p_len_orig = len(p_ids)

            # Load teacher reply ids + logits
            teacher_path = f.get("teacher_logits_path", "")
            if not teacher_path or not os.path.exists(teacher_path):
                # No teacher data for this example → no KD here
                reply_ids = []
                reply_logits = np.zeros((0, vocab_size), dtype=np.float32)
            else:
                data = np.load(teacher_path)
                reply_ids = data["reply_ids"].astype(np.int64).tolist()
                reply_logits = data["logits"].astype(np.float32)
            r_len_orig = len(reply_ids)

            if r_len_orig == 0:
                # No reply to distill: keep as much prompt as fits
                if p_len_orig > self.max_length:
                    p_ids_eff = p_ids[-self.max_length:]  # keep last tokens
                else:
                    p_ids_eff = p_ids
                r_ids_eff = []
                r_logits_eff = np.zeros((0, vocab_size), dtype=np.float32)
            else:
                # We want: p_len_eff + r_len_eff <= max_length
                # And at least 1 prompt token to predict the first reply token
                max_len = self.max_length
                max_r_len = min(r_len_orig, max_len - 1)
                max_p_len = max_len - max_r_len  # >= 1

                # Keep only the last max_p_len prompt tokens (truncate from the left)
                if p_len_orig <= max_p_len:
                    p_ids_eff = p_ids
                else:
                    p_ids_eff = p_ids[-max_p_len:]

                # Keep the first max_r_len reply tokens
                r_ids_eff = reply_ids[:max_r_len]
                r_logits_eff = reply_logits[:max_r_len]

            prompt_ids_list.append(p_ids_eff)
            reply_ids_list.append(r_ids_eff)
            reply_logits_list.append(r_logits_eff)
            prompt_lens.append(len(p_ids_eff))
            reply_lens.append(len(r_ids_eff))

        # ---- Build combined sequences ----
        combined_ids_list = [
            prompt_ids_list[i] + reply_ids_list[i] for i in range(batch_size)
        ]
        max_combined_len = max(len(seq) for seq in combined_ids_list)

        pad_id = self.tokenizer.pad_token_id
        padded_input_ids = torch.full(
            (batch_size, max_combined_len),
            fill_value=pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_combined_len), dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_combined_len),
            fill_value=-100,
            dtype=torch.long,
        )
        teacher_logits_tensor = torch.zeros(
            (batch_size, max_combined_len, vocab_size),
            dtype=torch.float32,
        )

        # ---- Align labels + teacher logits with student time steps ----
        for i in range(batch_size):
            seq = combined_ids_list[i]
            L = len(seq)
            padded_input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :L] = 1

            p_len = prompt_lens[i]   # effective prompt length AFTER truncation
            r_len = reply_lens[i]    # effective reply length kept

            if r_len > 0:
                # Logit at index (p_len - 1) predicts the first reply token
                start_idx = p_len - 1
                end_idx = start_idx + r_len   # by construction, <= L <= max_combined_len

                # Labels: only for reply tokens (prompt positions stay -100)
                labels[i, start_idx:end_idx] = torch.tensor(
                    reply_ids_list[i], dtype=torch.long
                )

                # Teacher logits: same time steps as labels
                teacher_logits_tensor[i, start_idx:end_idx, :] = torch.tensor(
                    reply_logits_list[i], dtype=torch.float32
                )

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "teacher_logits": teacher_logits_tensor,
        }


# ---------- KD loss function ----------
def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Token-wise KL divergence between student_log_probs and teacher_probs
    student_logits and teacher_logits: (B, S, V)
    mask: (B, S) where True indicates reply positions (compute KD there)
    """
    # Resolve device
    dev = "cuda"

    # No teacher or no positions to distill → 0 loss
    if teacher_logits is None or mask is None:
        return torch.tensor(0.0, device=dev)

    if teacher_logits.numel() == 0:
        return torch.tensor(0.0, device=dev)

    t = temperature

    # Student keeps gradient
    student_logp = F.log_softmax(student_logits / t, dim=-1)
    # Teacher is treated as fixed target
    with torch.no_grad():
        teacher_p = F.softmax(teacher_logits / t, dim=-1)

    b, s, v = student_logp.shape
    student_flat = student_logp.view(-1, v)
    teacher_flat = teacher_p.view(-1, v)
    mask_flat = mask.view(-1).bool()

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=dev)

    student_sel = student_flat[mask_flat]   # (N, V)
    teacher_sel = teacher_flat[mask_flat]   # (N, V)

    loss = F.kl_div(student_sel, teacher_sel, reduction="batchmean")
    return loss * (t * t)


class KDTrainer(Trainer):
    """
    Custom Trainer that combines KD loss and optional CE loss on the same tokens.
    """
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        # Pull out our custom fields (already on correct device thanks to Trainer)
        teacher_logits = inputs.pop("teacher_logits", None)
        labels = inputs.pop("labels", None)

        # Forward pass
        outputs = model(**inputs)
        student_logits = outputs.logits  # (B, S, V)
        current_device = student_logits.device

        # Move teacher logits to same device & dtype
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(device=current_device, dtype=torch.float32)

        # Build mask from labels we just popped (NOT from inputs again)
        reply_mask = (labels != -100) if labels is not None else None

        # KD loss on reply positions
        kd_l = kd_loss_fn(
            student_logits,
            teacher_logits,
            mask=reply_mask,
            temperature=TEMP,
            device=current_device,
        )

        # Optional CE loss on the same positions
        ce_l = torch.tensor(0.0, device=current_device)
        if (
            labels is not None
            and CE_WEIGHT > 0.0
            and reply_mask is not None
            and reply_mask.any()
        ):
            B, S, V = student_logits.shape
            logits_flat = student_logits.view(-1, V)
            labels_flat = labels.view(-1)
            mask_flat = reply_mask.view(-1).bool()

            logits_sel = logits_flat[mask_flat]  # logits only where labels are real
            labels_sel = labels_flat[mask_flat]

            ce_l = F.cross_entropy(logits_sel, labels_sel)

        loss = KD_WEIGHT * kd_l + CE_WEIGHT * ce_l


        # ---------- Log completion_length to wandb ----------
        if reply_mask is not None and reply_mask.any() and wandb.run is not None:
            # number of answer tokens per sample
            completion_lengths = reply_mask.sum(dim=-1).float()  # (batch,)
            mean_len = completion_lengths.mean().item()
            wandb.log({"train/completion_length": mean_len}, commit=False)


        if return_outputs:
            return loss, outputs
        return loss



# ---------- Setup trainer & training args ----------
tokenizer = tokenizer  # already instantiated above
data_collator = DistillDataCollator(tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=200,
    fp16=False,
    # fp16=FP16,
    remove_unused_columns=False,  # important because we use custom fields
    report_to="none",  # disable logging to wandb
)

# instantiate trainer
trainer = KDTrainer(
    model=student_model,
    args=training_args,
    train_dataset=student_prepared,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
print("Trainable params count:", sum(p.numel() for p in student_model.parameters() if p.requires_grad))


# Quick sanity batch check
sample_batch = data_collator([student_prepared[i] for i in range(min(2, len(student_prepared)))])
print("sample input_ids dtype/device:", sample_batch["input_ids"].dtype, sample_batch["input_ids"].device)
print("sample teacher_logits shape/dtype:", sample_batch["teacher_logits"].shape, sample_batch["teacher_logits"].dtype)
# Move to device like HF Trainer will
for k, v in sample_batch.items():
    sample_batch[k] = v.to(device)
# Forward a quick pass
with torch.no_grad():
    out = student_model(input_ids=sample_batch["input_ids"], attention_mask=sample_batch["attention_mask"])
print("student_logits dtype:", out.logits.dtype, "device:", out.logits.device)


# ---------- Run training ----------
trainer.train()

# save model/adapter
trainer.save_model(Path(OUTPUT_DIR) / "final")
tokenizer.save_pretrained(Path(OUTPUT_DIR) / "final")

print("Finished training. Artifacts saved to", OUTPUT_DIR)