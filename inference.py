from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text):
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("./final_model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Summarize
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
