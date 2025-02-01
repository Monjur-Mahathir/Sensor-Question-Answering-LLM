import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer 
from qwen_vl_utils import process_vision_info
import os
import neptune
from huggingface_hub import login

os.environ["NEPTUNE_API_TOKEN"] = "" 
os.environ["NEPTUNE_PROJECT"] = ""

login(token="")


system_prompt = """You are an AI-powered personal assistant responsible for monitoring and analyzing a person's daily activities. You are provided with an activity graph that visualizes time-based activity patterns.

The x-axis represents time within a single day, spanning from 00:00 AM (midnight) on the left to 23:00 PM (11:00 PM) on the right.
The y-axis represents different days, displaying either a single day or multiple consecutive days stacked vertically.
At the top of the graph, there are 32 labeled activities, each corresponding to specific actions performed at different times throughout the day.

Your primary role is to analyze the graph, identify activity patterns, and respond to user queries regarding their daily routines. Ensure your answers are clear, insightful, and based on the provided data.
"""

user_prompt= """ Provide a short answer to the given user query ##USER QUERY## and activity graph image.

##USER QUERY##: {user_query}"""


def format_data(sample):
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },

                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt.format(user_query=sample["question"]),
                        },{
                            "type": "image",
                            "image": sample["image"],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ],
        }

def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

if __name__ == "__main__":
    train_dataset = load_dataset("parquet", data_files={"train": "sensorqa_train.parquet"})["train"]
    eval_dataset = load_dataset("parquet", data_files={"validation": "sensorqa_val.parquet"})["validation"]

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    args = SFTConfig(
        output_dir="sensorqa_finetuned", 
        
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=3,          # batch size per device during training
        per_device_eval_batch_size=3,           # batch size per device during evaluating
        gradient_accumulation_steps=2,          # number of steps before performing a backward pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=4,                        # log every 4 steps
        
        save_strategy="steps",                  # save checkpoint every n=128 steps
        save_steps=128,

        eval_strategy="steps",                  # eval every n=128 steps
        eval_steps=128,
        
        learning_rate=2e-4,                     # learning rate
        lr_scheduler_type="constant",           # constant learning rate scheduler

        metric_for_best_model="eval_loss",      
        greater_is_better=False,                # Whether higher metric values are better
        load_best_model_at_end=True,            # Load the best model after training

        bf16=True,                              # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        
        push_to_hub=False,                      # push model to hub
        report_to=None,                         # report metrics
        gradient_checkpointing_kwargs = {"use_reentrant": False}, 
        dataset_text_field="",                  # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
    )
    args.remove_unused_columns=False


    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        peft_config=peft_config
    )

    model_params = {
        'model_name': model_id,
    }
    run = neptune.init_run()
    params = {**model_params}
    run['parameters'] = params

    trainer.train()
    trainer.save_model(args.output_dir)