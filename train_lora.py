import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import numpy as np

# Initialize model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def load_dialogue_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    conversations = []
    current_conversation = []
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            
            # Format the conversation with special tokens
            formatted_text = f"Interviewer: {question}\nTrump: {answer}"
            conversations.append({"text": formatted_text})
    
    return Dataset.from_list(conversations)

def tokenize_function(examples):
    # Tokenize the texts with padding and truncation
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def train():
    # Load and prepare the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Load and preprocess the dataset
    dataset = load_dialogue_dataset("trump-dialogue.txt")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="trump_lora_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=3,
    )
    
    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model.save_pretrained("trump_lora_model/final")
    
if __name__ == "__main__":
    train() 