import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
import os
import evaluate
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from huggingface_hub import login

# Ensure nltk resources are available
nltk.download("wordnet")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

os.environ["NEPTUNE_API_TOKEN"] = "" 
os.environ["NEPTUNE_PROJECT"] = "" 
login(token="")


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

system_prompt = """You are an AI-powered personal assistant responsible for monitoring and analyzing a person's daily activities. You are provided with an activity graph that visualizes time-based activity patterns.

The x-axis represents time within a single day, spanning from 00:00 AM (midnight) on the left to 23:00 PM (11:00 PM) on the right.
The y-axis represents different days, displaying either a single day or multiple consecutive days stacked vertically.
At the top of the graph, there are 32 labeled activities, each corresponding to specific actions performed at different times throughout the day.

Your primary role is to analyze the graph, identify activity patterns, and respond to user queries regarding their daily routines. Ensure your answers are clear, insightful, and based on the provided data.
"""
prompt= """ Provide a short answer to the given user query ##USER QUERY## and activity graph image.

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
                            "text": prompt.format(user_query=sample["question"]),
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


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    response_gt = sample["messages"][2]['content'][0]['text']
    #print(f"OK\nThe original reponse: {response_gt}\nOK\n")
    
    text_input = processor.apply_chat_template(sample["messages"][:2], tokenize=False)
    image_inputs = process_vision_info(sample["messages"][:2])[0]
    
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response_pred = output_text[0]#.split("\n")[-1]
    response_pred = response_pred.replace("\n", "")
    response_pred = response_pred.replace("assistant", "")
    response_pred = response_pred.replace("answer", "")
    #print(f"OK\nThe predicted reponse: {response_pred}\nOK\n")

    return response_gt, response_pred

if __name__ == "__main__":
    eval_dataset = load_dataset("parquet", data_files={"validation": "sensorqa_val.parquet"})["validation"]
    eval_dataset = [format_data(sample) for sample in eval_dataset]

    eval_dataset_short = load_dataset("parquet", data_files={"validation": "sensorqa_val.parquet"})["validation"]
    eval_dataset_short = [format_data(sample) for sample in eval_dataset_short]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
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

    # Apply PEFT model adaptation
    model = get_peft_model(model, peft_config)
    adapter_path = "weight/checkpoint-1664"
    adapter_name = "sensorqa_adapter"  
    model.load_adapter(adapter_path, adapter_name)
    model.set_adapter(adapter_name)

    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0

    for i in range(len(eval_dataset)):
        response_gt, response_pred = generate_text_from_sample(model, processor, eval_dataset[i])

        references = [response_gt]
        predictions = [response_pred]

        # Compute ROUGE scores
        rouge_scores = rouge_metric.compute(predictions=predictions, references=references)

        # Compute BLEU using nltk (with smoothing)
        smooth = SmoothingFunction().method1
        nltk_bleu = sentence_bleu([response_gt.split()], response_pred.split(), smoothing_function=smooth)

        rouge_1 += torch.tensor(rouge_scores["rouge1"]).item()
        rouge_2 += torch.tensor(rouge_scores["rouge2"]).item()
        rouge_l += torch.tensor(rouge_scores["rougeL"]).item()
        bleu += torch.tensor(nltk_bleu).item()

        line = f"####Ground Truth:\n{response_gt}\n####LLM Response:\n{response_pred}\n####Scores:\n{torch.tensor(rouge_scores["rouge1"]).item()} {torch.tensor(rouge_scores["rouge2"]).item()} {torch.tensor(rouge_scores["rougeL"]).item()} {torch.tensor(nltk_bleu).item()}\n\n\n"
        with open('results.txt', 'a') as f:
            f.writelines(line)
        f.close()

    print(f"Average Rouge 1: {rouge_1 / i:.2f}") #0.71
    print(f"Average Rouge 2: {rouge_2 / i:.2f}") #0.55
    print(f"Average Rouge L: {rouge_l / i:.2f}") #0.69
    print(f"Average Bleu: {bleu / i:.2f}") #0.44

        