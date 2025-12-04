import re
import json
import random
import torch
import numpy as np

from unsloth import FastLanguageModel
from datasets import load_dataset
from collections import Counter
from datasets import Dataset as HFDataset
from pathlib import Path
from tqdm import tqdm


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
    load_in_4bit = False,    
    load_in_8bit = False,    
    full_finetuning = False, 
    # token = "hf_...",      # use one if using gated models
)

# LoRA adapter for teacher model
teacher_model = FastLanguageModel.get_peft_model(
    base_model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # rank stabilized LoRA
    loftq_config = None,  # LoftQ
)
teacher_model.to(model_dtype)


#Finds the last \\boxed{...} answer in a string
def extract_boxed_answer(text):
    last = None
    for m in re.finditer(r'\\boxed{', text):
        i = m.end()       # position right after the opening '{'
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:    # found the matching closing brace
            last = text[start:i-1]
    return last


#### Load and prep the dataset
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
math_dataset = load_dataset("openai/gsm8k", "main", split="train")

idx = 0 # starting index for repeated dataset creation
min_samples = 6
max_samples = min(min_samples,len(math_dataset)) # max number of samples in the created math dataset for ssb
questions = math_dataset["question"]
answers = [extract_hash_answer(answer) for answer in math_dataset["answer"]]
max_examples = len(questions) # maximus size of the dataset
num_trials = 4
teacher_data = []
student_data = []

# Generates training data by finding correct and incorrect answers from model responses
def generate_ssb_training_data(model_used):
    global teacher_data
    global student_data
    # idx = 0
    global idx
    sample_num = 0
    inf_prompt = "**Role:** You are an expert math tutor. When you are given a problem to solve, you provide detailed step-by-step reasoning in your solution. Your response is clear, precise, and unambiguous. You do not skip any step. You put your final answer within \\boxed{} at the end of your response."

    while len(teacher_data) < max_samples and idx < max_examples:
        print(f"--- Processing Question #{idx+1} ---")
        rollout_responses = []
        # Step 1: Perform rollouts to collect a variety of answers
        for i in range(num_trials):
            print(f"  > Rollout {i+1}/{num_trials} for answering question #{idx+1}...")
            try:

                # LLM generation
                messages = [
                    {"role": "system", "content": inf_prompt},
                    {"role" : "user", "content" : questions[idx]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize = False,
                    add_generation_prompt = True, 
                )
                inputs = tokenizer(text=text, return_tensors = "pt").to("cuda")
                with torch.no_grad():
                    outputs = model_used.generate(
                        **inputs,
                        max_new_tokens=8192,
                        output_scores = True,          
                        return_dict_in_generate = True, 
                        # --- KEY CHANGES ---
                        do_sample = True,  
                        top_k = 50,        
                        temperature = 0.8,
                    )
                # The generated text
                generated_text = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                # print("  > Generated Text:\n", generated_text)

                # Post processing
                model_answer = extract_boxed_answer(generated_text)
                rollout_responses.append({
                    "answer": model_answer,
                    "generation": generated_text
                })
            except Exception as e:
                print(f"    An error occurred: {e}")
                rollout_responses.append({"answer": None, "generation": ""}) # Log failure

        # Step 2: Analyze the collected answers
        correct_rollouts = [r for r in rollout_responses if r["answer"] == answers[idx]]
        incorrect_rollouts = [r for r in rollout_responses if r["answer"] is not None and r["answer"] != answers[idx]]

        # Step 3: Conditional logic to decide whether to generate a ssb pair
        if not correct_rollouts or not incorrect_rollouts:
            print(f"  Skipping question: Not a mix of correct and incorrect answers (Correct: {len(correct_rollouts)}, Incorrect: {len(incorrect_rollouts)}).")
            idx += 1
            continue

        # Step 4: Find the majority wrong answer, or a random one if no majority exists.
        incorrect_answer_counts = Counter(r["answer"] for r in incorrect_rollouts)
        most_common_wrong = incorrect_answer_counts.most_common(1)[0]

        if most_common_wrong[1] > 1:
             # There is a clear majority wrong answer
            selected_wrong_answer = most_common_wrong[0]
            print(f"  Found a mix. Majority Wrong Answer: {selected_wrong_answer}")
        else:
            # All incorrect answers are unique, pick one randomly
            selected_wrong_answer = random.choice(list(incorrect_answer_counts.keys()))
            print(f"  Found a mix. No majority, randomly selected Wrong Answer: {selected_wrong_answer}")

        # Get the full generation for the chosen correct response and the rejected incorrect one
        chosen_correct_generation = correct_rollouts[0]["generation"]
        rejected_generation = next((r["generation"] for r in incorrect_rollouts if r["answer"] == selected_wrong_answer), None)

        print("  Generating a robust, final answer...")

        # Step 5: Generate the robust, "chosen" response using context from a correct and incorrect rollout
        try:
            sys_prompt = """**Role:** You are an expert math tutor. When you are given a problem to solve, you provide detailed step-by-step reasoning in your solution. Your response is clear, precise, and unambiguous. You do not skip any step. You put your final answer within \\boxed{} at the end of your response.
            """

            ssb_prompt = f"""We have two sample responses from students for a math problem for you to observe. One of the responses is correct and the other is incorrect. The students were asked to put the final answer inside boxed in their responses. Only this final answer was checked by an automatic evaluator. The following is the problem:
            ---
            {questions[idx]}
            ---

            This is the correct response from a student:
            ---
            {chosen_correct_generation}
            ---

            This is an attempt by another student which was labelled as incorrect by auto-evaluator:
            ---
            {rejected_generation}
            ---

            Write a coherent, step-by-step derivation of the solution. Do not skip any step in your response and make it as detailed as posible. This should be a standalone solution of the given problem since this solution will be used by the students for studying and learning. Make the solution robust by cautioning about potential errors or wrong chain of reasoning. However, do not mention in your response that you were provided attempted responses by the students. There should not be a slightest mention or hint that you are actually refining from correct or incorrect responses written by students.

            Conclude the entire response with the final answer, enclosed in boxed, at the very end. Make sure your enclosed final answer exactly matches the enclosed final answer in the given correct response of the student.
            """

            # LLM generation
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role" : "user", "content" : ssb_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True, # Must add for generation
            )
            inputs = tokenizer(text=text, return_tensors = "pt").to("cuda")
            with torch.no_grad():
                outputs = model_used.generate(
                    **inputs,
                    max_new_tokens = 8192,
                    output_scores = True,          
                    return_dict_in_generate = True, 
                    do_sample = True,  
                    top_k = 50,        
                    temperature = 0.4,
                )
            # The generated text
            final_generation = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


            # Final check to ensure the robust answer is actually correct
            final_robust_answer = extract_boxed_answer(final_generation)
            if final_robust_answer == answers[idx]:
                # Save the actual hinted prompt that was used in teacher dataset
                teacher_data.append({
                    "messages": [
                        {"role": "system", "content": inf_prompt},
                        {"role" : "user", "content" : ssb_prompt},
                        {"role": "assistant", "content": final_generation} # Use the full, correct generation
                    ]
                })
                # Save the question only in student dataset
                student_data.append({
                    "messages": [
                        {"role": "system", "content": inf_prompt},
                        {"role": "user", "content": questions[idx]},
                    ]
                })
                sample_num += 1
                print(f"  Successfully generated and saved a ssb pair of entry number {sample_num}.")
            else:
                print(f"  Skipping: The generated robust answer was incorrect. Expected {answers[idx]}, got {final_robust_answer}")

        except Exception as e:
            print(f"    An error occurred during the robust answer generation: {e}")

        idx += 1


    teacher_output_filename = "generated_math_teacher_data.jsonl"
    with open(teacher_output_filename, "w") as f:
        for entry in teacher_data:
            f.write(json.dumps(entry) + "\n")

    student_output_filename = "generated_math_student_data.jsonl"
    with open(student_output_filename, "w") as f:
        for entry in student_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\n--- Finished ---")
    print(f"Saved {len(teacher_data)} ssb examples to {teacher_output_filename}")
    return teacher_output_filename

generate_ssb_training_data(teacher_model)

teacher_dataset = HFDataset.from_list(teacher_data)
student_dataset = HFDataset.from_list(student_data)

teacher_logits_dir = Path("teacher_logits")
teacher_logits_dir.mkdir(exist_ok=True)

# ---------- Helper functions ----------
def flatten_messages_to_text(messages):
    """
    messages: list of dicts with 'role' and 'content'
    returns a single string that concatenates role markers and content. Adjust style as needed.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append("User: " + content)
        elif role == "assistant":
            parts.append("Assistant: " + content)
        else:
            parts.append(f"{role.capitalize()}: " + content)
    # Separate with two newlines to help tokenizers with structure
    return "\n\n".join(parts)

def make_prompt_and_reply_from_teacher_messages(messages):
    """Return (prompt_text, reply_text). We assume teacher messages format: user then assistant reply."""
    # find first user block(s) as prompt and the assistant reply as reply
    user_parts = []
    reply_parts = []
    for m in messages:
        if m["role"] == "user":
            user_parts.append(m["content"])
        elif m["role"] == "assistant":
            reply_parts.append(m["content"])
    prompt_text = " ".join(user_parts).strip()
    reply_text = " ".join(reply_parts).strip()
    return prompt_text, reply_text

def forward_model(model, input_ids, attention_mask=None):
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    # try variety of call signatures
    try:
        out = model(**kwargs)
    except Exception:
        try:
            out = model.model(**kwargs)
        except Exception as e:
            raise RuntimeError("Error occured.") from e
    return out


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 8192

# ---------- Precompute teacher reply logits and save per-example ----------
print("Precomputing teacher logits and saving to", teacher_logits_dir)
teacher_index = {}
vocab_size = teacher_model.model.lm_head.out_features
print(f"Using model's actual vocab size: {vocab_size}")

for i, item in enumerate(tqdm(teacher_dataset)):
    messages = item["messages"] # this is a list-of-dicts representing conversation

    # Extract prompt/reply from chat structure
    if not messages or messages[-1]["role"] != "assistant":
        print(f"Warning: teacher example {i} has no assistant reply. Skipping.")
        continue

    prompt_messages = messages[:-1]
    teacher_reply_text = messages[-1]["content"]

    # Apply chat template to prompt (to match generation)
    teacher_prompt_chat_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True # Match the generation call
    )

    # Tokenize prompt
    prompt_enc = tokenizer(teacher_prompt_chat_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)

    # Fix 2: Tokenize reply WITHOUT special tokens (BOS/EOS)
    # This is critical to avoid a mis-aligned concatenation
    reply_enc = tokenizer(teacher_reply_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)

    # build combined inputs by concatenation
    input_ids = torch.cat([prompt_enc["input_ids"][0], reply_enc["input_ids"][0]], dim=0).unsqueeze(0).to(DEVICE)
    # Create matching attention mask
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    with torch.no_grad():
        out = forward_model(teacher_model, input_ids=input_ids, attention_mask=attention_mask)
        logits_tensor = out.logits.squeeze(0).detach().cpu()
        logits = logits_tensor.float().numpy()  # (seq_len, vocab)

    prompt_len = prompt_enc["input_ids"].shape[1]
    reply_len = reply_enc["input_ids"].shape[1]

    # Correct off-by-one error in logit slicing
    # Logits at position `k` predict token at position `k+1`.
    # We want the logits that predict the *reply tokens*.
    # The first reply token is at index `prompt_len`.
    # The logits predicting it are at index `prompt_len - 1`.
    reply_logits = logits[prompt_len - 1 : prompt_len + reply_len - 1, :]
    reply_ids = reply_enc["input_ids"].squeeze(0).cpu().numpy() # shape (reply_len,)

    # Ensure we sliced correctly
    if reply_logits.shape[0] != reply_len:
        print(f"Warning: Mismatch in reply/logit length for example {i}. Skipping.")
        continue

    # Save per-example
    fname = teacher_logits_dir / f"teacher_logits_{i}.npz"
    np.savez_compressed(fname, logits=reply_logits.astype(np.float32), reply_ids=reply_ids.astype(np.int32))
    teacher_index[str(i)] = str(fname)

# Save index mapping (optional)
with open(teacher_logits_dir / "index.json", "w") as f:
    json.dump(teacher_index, f)

print("Saved teacher logits for", len(teacher_index), "examples.")