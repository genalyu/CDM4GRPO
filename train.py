from unsloth import FastLanguageModel
import torch
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
max_seq_length = 2048# Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen2.5-3B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = False, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd

# Load and prep dataset
SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""
q_matrix = pd.read_csv("CDM/q_matrix.csv", index_col=0).values  
q_matrix = torch.tensor(q_matrix, dtype=torch.float32)  
def get_qa(split="train") -> Dataset:
    data = load_from_disk('data/openr1-1500samples')  
    data = data.map(lambda x, idx: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'exer_id': idx,  
        'knowledge_emb': q_matrix[idx].tolist(), 
        'reward': 0.1,  
    }, with_indices=True)  
    return data 

dataset = get_qa()

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

model_name = "RM-Gemma-2B"

rm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

def reward_func(prompts, completions, **kwargs) -> list[float]:

    questions = [prompt[-1]["content"] for prompt in prompts]
    responses = [completion[0]["content"] for completion in completions]

    scores = []
    for q, r in zip(questions, responses):
        inputs = rm_tokenizer(
            q, r,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        ).to(rm_model.device)

        with torch.no_grad():
            outputs = rm_model(**inputs)
            score = outputs.logits.squeeze().item()
            scores.append(score)
    
    return scores


import torch
import pandas as pd

k_difficulty = pd.read_csv("CDM/k_difficulty.csv").values  
k_difficulty = torch.tensor(k_difficulty, dtype=torch.float32)

def data_filter_fn(dataset, opt, alpha=0.05):
    def filter_condition(example):
        exer_id = torch.tensor(example["exer_id"])  
        kn_emb = torch.tensor(example["knowledge_emb"])  

        score = torch.tensor(example["reward"])  

        target_output = score.unsqueeze(0)  
        optimized_emb = opt.optimize_embedding(exer_id, kn_emb, target_output, steps=5)

        current_k_diff = k_difficulty[exer_id].to(device)
        nonzero_idx = (current_k_diff != 0).nonzero(as_tuple=True)[0]
        idx = nonzero_idx[0]
        
        diff = torch.abs(current_k_diff[idx] - optimized_emb[idx])
        return diff.item() < alpha

    return dataset.filter(filter_condition)

max_prompt_length = 800

from trl import GRPOConfig, GRPOTrainer
import os
os.environ["WANDB_MODE"] = "offline"  # Run offline
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory 
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)

from CDM.train import StudentEmbeddingOptimizer
from CDM.model import Net
CDM_path= "CDM/model/model_epoch5" 
student_n = 7
exer_n = 1500
knowledge_n = 8
q_matrix = pd.read_csv("CDM/q_matrix.csv",  index_col=0).values  
q_matrix = torch.tensor(q_matrix, dtype=torch.float32)
CDM = Net(student_n, exer_n, knowledge_n, q_matrix).to(device)
CDM.load_state_dict(torch.load(CDM_path))
opt = StudentEmbeddingOptimizer(CDM)

from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer
trainer = UnslothGRPOTrainer(
    train_dataset = dataset,
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        reward_func,
    ],
    opt=opt,
    data_filter_fn=lambda ds, opt: data_filter_fn(ds, opt),
    args = training_args,
)
trainer.train()