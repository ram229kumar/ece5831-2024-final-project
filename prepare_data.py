from datasets import load_dataset
from transformers import BartTokenizer

def prepare_data():
    # Load CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0",split={
        "train": "train[:3%]",        # Use 5% of the training set
        "validation": "validation[:3%]",  # Use 5% of the validation set
        "test": "test[:3%]"           # Use 5% of the test set
    })
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["highlights"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets.save_to_disk("tokenized_cnn_dailymail")

if __name__ == "__main__":
    prepare_data()
