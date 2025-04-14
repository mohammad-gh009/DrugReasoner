import pandas as pd
from datasets import Dataset
import re
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer

df = pd.read_csv("/home/u111169/mgh/train_reason_main.csv" )

df["labels"].replace({"<APPROVED>":"approved" , "<NOT APPROVED>":"unapproved"} , inplace=True)



max_seq_length = 5000 
lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/u111169/wrkdir/mgh-project/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    max_seq_length = max_seq_length,
    load_in_4bit = True, 
    fast_inference = True, 
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
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



SYSTEM_PROMPT = """
You are a **chemist specializing in drug discovery and molecular modeling**, tasked with analyzing chemical compounds for drug-likeness and viability.
Your goal is to determine the compound’s potential as a drug candidate by integrating these computational criteria.
think step by step and reason whether the compound will be approved or  unapproved.
then label your final decession with approved and unapproved. 
also provide a confident score for your final label based on your thinking(the confident score is between [0 , 1] range).
Return a score close to 1 if the compound appears strongly approved or unapproved. If you're uncertain, return a score closer to 0, scaling it based on your level of uncertainty—the greater the uncertainty, the closer the score should be to zero.
Respond in the following format:
<think>
...
</think>
<label>
...
</label>
<score>
...
</score>
"""

XML_COT_FORMAT = """\
<think>
{think}
</think>
<label>
{label}
</label>
<score>
{score}
</score>
"""


def extract_xml_label(text: str) -> str:
    answer = text.split("<label>")[-1]
    answer = answer.split("</label>")[0]
    return answer.strip()

def extract_xml_score(text: str) -> str:
    answer = text.split("<score>")[-1]
    answer = answer.split("</score>")[0]
    return answer.strip()

# uncomment middle messages for 1-shot prompting
def get_dataset(split = "train") -> Dataset:
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"""
            I have developed a model that can predict the drug approval of compound X.
            This model outputs two lists containting the properties of most similar approved and unapproved modelcules
            Your task is to analyze the likelihood of compound X receiving regulatory approval based on its similarity to known molecules.

            - RDKit Analysis of Compound X:
            {x["rdkit_info"]}

            Using the provided data:
            {x['Dicts']}
            """}
        ],
        'answer': x['labels']
    }) 
    return data
dataset = get_dataset()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_label(r).lower() for r in responses]
    extracted_responses_score = [extract_xml_score(r) for r in responses]
    print(f"{'-'*20}\nQuestion:\n{q}\n\n\nAnswer\n\n\n:\n{answer[0]}\n\n\nResponse\n\n\n:\n{responses[0]}\n\n\nExtracted\n\n\n:\n{extracted_responses[0]}\n\n\nScore\n\n\n:\n{extracted_responses_score[0]}")
    return [3.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_label(r).lower() for r in responses]
    return [0.5 if r in ["approved", "unapproved"] else 0.0 for r in extracted_responses]
    
def int_score_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_score(r) for r in responses]
    return [0.5 if isinstance(r, (int, float)) and 0 <= r <= 1 else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<label>\n.*?\n</label>\n<score>\n.*?\n</score>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<label>.*?</label>\s*<score>.*?</score>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<label>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</label>\n")[-1])*0.001
    if text.count("\n</label>") == 1:
        count += 0.125
        count -= (len(text.split("\n</label>")[-1]) - 1)*0.001
    if text.count("\n<score>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</score>\n")[-1])*0.001
    if text.count("\n</score>") == 1:
        count += 0.125
        count -= (len(text.split("\n</score>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def confident_score_func(completions, answer, **kwargs): 
    responses = [completion[0]['content'] for completion in completions]
    extracted_label = [extract_xml_label(r).lower() for r in responses]
    extracted_score = [extract_xml_score(r) for r in responses]
    
    count = 0.0
    for r, a, s in zip(extracted_label, answer, extracted_score):
        # Check if r is a float
        try:
            float_s = float(s)
        except ValueError:
            return [0.0]
            
        if r == a and float_s >= 0.7: 
            count += 5.0
        elif r == a and 0.7 > float_s >= 0.4: 
            count += 1.0
        elif r == a and 0.4 > float_s: 
            count += 0.0
        elif r != a and float_s >= 0.7: 
            count += 0.0
        elif r != a and 0.7 > float_s >= 0.4: 
            count += 1.0
        elif r != a and 0.4 > float_s: 
            count += 5.0
    return [count]

max_prompt_length = 2000


training_args = GRPOConfig(
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.01,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 4, # test 
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 20,
    save_steps = 1,
    max_grad_norm = 0.1,
    report_to = "none", 
    output_dir = "/home/u111169/mgh/checkpoints",
    #logging_dir = "/home/u111169/mgh/log"
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        int_score_reward_func,
        confident_score_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()