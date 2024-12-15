from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_custom_text(text, model_path="fine_tuned_bart", max_length=2000, min_length=90):
    """
    Generate a summary for the custom input text using the fine-tuned BART model.
    """
    # Load the fine-tuned model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    # Tokenize the input text
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    with open("custom_input.txt", "r") as file:
        text = file.read()
    
    # Generate and print the summary
    summary = summarize_custom_text(text)
    
    print("Summary:")
    print(summary)
