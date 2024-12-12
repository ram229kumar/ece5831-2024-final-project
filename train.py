import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load Pretrained Model and Tokenizer
model_name = "t5-small"  # You can choose "t5-base" or "t5-large" for better performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 2: Load Dataset
def preprocess_data(examples):
    """
    Preprocess data to tokenize input and target texts.
    """
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize summaries as target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"].map(preprocess_data, batched=True)
val_data = dataset["validation"].map(preprocess_data, batched=True)

# Step 3: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available()  # Use mixed precision if GPU is available
)

# Step 4: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Step 5: Train the Model
trainer.train()

# Step 6: Save the Fine-Tuned Model
model.save_pretrained("./custom_summarizer")
tokenizer.save_pretrained("./custom_summarizer")

print("Model fine-tuned and saved successfully!")
