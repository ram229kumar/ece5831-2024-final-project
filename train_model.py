from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk

def train_model():
    # Load tokenized dataset
    tokenized_datasets = load_from_disk("tokenized_cnn_dailymail")

    print("Train Columns: ",tokenized_datasets["train"].column_names)
    print("Validation Columns: ",tokenized_datasets["validation"].column_names)
    print("Test Columns: ",tokenized_datasets["test"].column_names)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # Define training arguments
    training_args = TrainingArguments(
        logging_steps=1000,
        output_dir="bart-summarizer",
        eval_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Small batch size for CPU
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Accumulate gradients to simulate a larger batch size
        num_train_epochs=1,
        save_steps=1000,
        save_total_limit=2,
        fp16=False,  # Disable mixed precision
    )


    # Define the Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use the data collator here
    )

    # Train the model
    trainer.train()
    model.save_pretrained("fine_tuned_bart")
    tokenizer.save_pretrained("fine_tuned_bart")

if __name__ == "__main__":
    train_model()
