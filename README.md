# ece5831-2024-final-project &#8594; Text Summarizer

This project demonstrates an abstractive text-summarization using a fine tuned BART Model on the CNN/DailyMail dataset. The project includes data preparation, model fine-tuning, evaluation and summarization of custom input text.

## Requirements

### Environment

Python 3.8 or Later is required.

### Dependencies

Install the libraries in `requirements.txt` or run the command `pip install -r requirements.txt`. It will automatically install all the required libraries.

### Hardware Requirements

- CPU: Supported but significantly slower for training.

- GPU Recommended: Training on Google Colab with GPU is highly recommended for faster processing.

## Files Overview

1. `prepare_data.py`

   This script loads and preprocesses the CNN/DailyMail dataset. This script downloads, tokenizes the dataset and saves the tokenized dataset to disk for later use.

2. `train_model.py`

   This script fine-tunes the BART model on the tokenized dataset. This script loads the tokenized dataset, fine-tunes the BART Model with specified training arguments and Finally saves the fine-tuned model and tokenizer.

3. `evaluate_model.py`

   This script evaluates the fine-tuned model on a small test subset. This script generates summaries for test articles and calculates their ROUGE scores to evaluate model performance.

4. `test_custom_input.py`

   This script generates a summary for a custom input text file. Loads the fine-tuned model, tokenizer and prints the generated summary for the provided `custom_input.txt`.

## How to run the Project

Run the `prepare_data.py` script to preprocess and save the tokenized dataset.

`python prepare_data.py`

Fine-tune the model using the `train_model.py` script.

`python train_model.py`

This will fine-tune the BART model using 3% of the CNN/DailyMail Dataset and Saves the model and tokenizer in the `fine_tuned_bart` directory (Won't be present in github due to its size. You can download it from the google drive)

Evaluate the model's performance using the `evaluate_model.py` script.

`python evaluate_model.py`

This will give the ROUGE Scores.

Example: `{'rouge1': 0.3264068795537438, 'rouge2': 0.13793284198542616, 'rougeL': 0.23947143286195077, 'rougeLsum': 0.3034078667048914}`

Finally, if one wants to test the code you can modify or replace the text in `custom_input.txt` and run the below code.

`python test_custom_input.py`

## Project Workflow

1. Data Preparation: Preprocess and tokenize the CNN/DailyMail dataset.

2. Model Training: Fine-tune the BART model on the tokenized dataset.

3. Evaluation: Evaluate the model’s performance using ROUGE metrics.

4. Inference: Generate summaries for custom text inputs.

## Output

The below screenshot is of the model successfully getting trained and getting evaluated and displaying ROUGE Scores.

![Output Image](</Output Image.jpg>)

The below screenshot is the output of the model running with the custom_input.txt as the input.

For the custom input available in `custom_input.txt`. The below image is the output.

![Output Image](/Custom_Inputs_Output.png)

## Notes

- Adjust the max_length and min_length in `test_custom_input.py` to control the summary size.

- Use Google Colab for faster training with GPU support. (Be cautious as if the window is inactive for longer period it will stop the process abruptly).

- Ensure the fine_tuned_bart directory exists after training to run evaluation and inference scripts.

## Links

Google Drive (Has code and Model trained with 3% dataset): [Drive](https://drive.google.com/drive/folders/1HlXyx0RUPqoRFnxRyRuGxfQ9mojKpChP?usp=sharing)

Project Report link : [Report](/Text_Summarizer_Project_Report.pdf)

Demo Video [Youtube](PLACEHOLDER.com)

## Future Works

- Train on a larger portion of the dataset.
- Use a better `facebook/bart-large` model instead of `facebook/bart-base` model.
- Experiment with different pre-trained models (Ex: Pegasus).
- Optimize hyperparameters for better ROUGE Scores.
