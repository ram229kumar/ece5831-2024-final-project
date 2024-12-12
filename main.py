from inference import summarize_text

article = "The Transformers library provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages."
summary = summarize_text(article)
print("Summary:", summary)
