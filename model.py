# Import the necessary libraries for AI and logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

# Set up basic logging to see informational messages in the terminal
logging.basicConfig(level=logging.INFO)

# Use a try-except block to handle potential errors during model loading
try:
    # Load the pre-trained sentiment analysis model from Hugging Face
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
# If loading fails, log the error and set the analyzer to None
except Exception as e:
    logging.error(f"Error loading Hugging Face sentiment model: {e}")
    sentiment_analyzer = None

# Define the function for local sentiment analysis
def analyze_sentiment_local(prompt: str) -> str:
    """
    Analyzes sentiment using a powerful local RoBERTa model.
    """
    # Check if the sentiment model failed to load
    if not sentiment_analyzer:
        # Raise an error if the model is not available
        raise ConnectionError("Sentiment analysis model could not be loaded. Please ensure you have an internet connection for the first run.")
        
    # Try to perform sentiment analysis on the user's prompt
    try:
        # Pass the prompt to the sentiment analysis pipeline
        result = sentiment_analyzer(prompt)
        # The model returns 'positive', 'negative', or 'neutral'
        return result[0]['label'].lower()
    # If analysis fails, raise a new error with a descriptive message
    except Exception as e:
        raise RuntimeError(f"Error during sentiment analysis: {e}")

# Use a try-except block for loading the larger text generation model
try:
    # Log that the text generation model is starting to load
    logging.info("Loading text generation model (TinyLlama/TinyLlama-1.1B-Chat-v1.0)... This will take time and RAM.")
    
    # Load the pre-trained language model for text generation
    generation_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto", # Automatically select the best data type (e.g., float16)
        device_map="auto", # Automatically use GPU if available, otherwise CPU
    )
    # Load the corresponding tokenizer for the language model
    generation_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Set the padding token to the end-of-sentence token if it's not already set
    if generation_tokenizer.pad_token is None:
        generation_tokenizer.pad_token = generation_tokenizer.eos_token

    # Log that the model has been loaded successfully
    logging.info("Text generation model loaded successfully.")
# If loading fails, log a fatal error and set model variables to None
except Exception as e:
    logging.error(f"FATAL: Could not load the text generation model. Your machine might not have enough RAM. Error: {e}")
    generation_model = None
    generation_tokenizer = None

# Define the function for local text generation
def generate_text_local(prompt: str, sentiment: str, output_length: str) -> str:
    """
    Generates text using a local instruction-tuned model (TinyLlama) with a robust, direct method.
    """
    # Check if the generation model or tokenizer failed to load
    if not generation_model or not generation_tokenizer:
        # Raise an error explaining the likely cause (e.g., not enough RAM)
        raise RuntimeError(
            "The text generation model could not be loaded. "
            "This often happens if your computer does not have enough RAM. "
            "Please check the terminal for specific error messages."
        )

    # Set the maximum number of new tokens based on the desired output length
    max_tokens = 200 if output_length == 'Paragraph' else 450
    
    # Create a structured message list for the chat model
    messages = [
        {"role": "system", "content": "You are a skilled writer."},
        {"role": "user", "content": f"Your task is to write a high-quality {output_length.lower()} that has a strong **{sentiment}** tone. The topic is: \"{prompt}\". Provide only the {output_length.lower()}, with no extra commentary or introduction."},
    ]
    
    # Apply the chat template to format the messages correctly for the model
    prompt_for_model = generation_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Use a try-except block for the text generation process
    try:
        # Convert the formatted prompt string into tokens and move to the correct device (CPU/GPU)
        inputs = generation_tokenizer(prompt_for_model, return_tensors="pt").to(generation_model.device)

        # Call the model's generate function with specific parameters
        outputs = generation_model.generate(
            **inputs,
            max_new_tokens=max_tokens, # Limit the length of the generated text
            eos_token_id=generation_tokenizer.eos_token_id, # Stop generation at the end-of-sentence token
            do_sample=True, # Enable creative, less deterministic output
            temperature=0.7, # Control the randomness of the output (lower is more predictable)
            top_p=0.9, # Use nucleus sampling to control the vocabulary diversity
        )

        # Decode the generated tokens back into a human-readable string, skipping the prompt
        response_only = generation_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Return the cleaned-up generated text
        return response_only.strip()

    # If generation fails, log detailed error info and raise a runtime error
    except Exception as e:
        logging.error(f"Error details during text generation: {e}", exc_info=True)
        raise RuntimeError(f"An error occurred during text generation: {e}")

