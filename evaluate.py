import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_metric

def evaluate():
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("./final_model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    # Load ROUGE metric
    rouge = load_metric("rouge")

    # Evaluate
    for example in dataset.select(range(100)):  # Evaluate on a subset
        inputs = tokenizer("summarize: " + example["article"], return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        pred_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rouge.add(prediction=pred_summary, reference=example["highlights"])

    # Compute and print ROUGE
    scores = rouge.compute()
    print(scores)

if __name__ == "__main__":
    evaluate()
