from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import evaluate

def evaluate_model():
    # Load model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("fine_tuned_bart")
    tokenizer = BartTokenizer.from_pretrained("fine_tuned_bart")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
    rouge = evaluate.load("rouge")

    summaries = []
    references = []

    for item in dataset:
        inputs = tokenizer(item["article"], max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0)
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        references.append(item["highlights"])

    scores = rouge.compute(predictions=summaries, references=references)
    print(scores)

if __name__ == "__main__":
    evaluate_model()
