import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)

try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
except Exception as e:
    logging.error(f"Error loading Hugging Face sentiment model: {e}")
    sentiment_analyzer = None

def analyze_sentiment_local(prompt: str) -> str:
    """
    Analyzes sentiment using a powerful local RoBERTa model.
    """
    if not sentiment_analyzer:
        raise ConnectionError("Sentiment analysis model could not be loaded. Please ensure you have an internet connection for the first run.")
        
    try:
        result = sentiment_analyzer(prompt)
        # Model returns 'positive', 'negative', or 'neutral'
        return result[0]['label'].lower()
    except Exception as e:
        raise RuntimeError(f"Error during sentiment analysis: {e}")

try:
    logging.info("Loading text generation model (TinyLlama/TinyLlama-1.1B-Chat-v1.0)... This will take time and RAM.")
    
    generation_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto", 
        device_map="auto",
    )
    generation_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    if generation_tokenizer.pad_token is None:
        generation_tokenizer.pad_token = generation_tokenizer.eos_token

    logging.info("Text generation model loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Could not load the text generation model. Your machine might not have enough RAM. Error: {e}")
    generation_model = None
    generation_tokenizer = None

def generate_text_local(prompt: str, sentiment: str, output_length: str) -> str:
    """
    Generates text using a local instruction-tuned model (TinyLlama) with a robust, direct method.
    """
    if not generation_model or not generation_tokenizer:
        raise RuntimeError(
            "The text generation model could not be loaded. "
            "This often happens if your computer does not have enough RAM. "
            "Please check the terminal for specific error messages."
        )

    max_tokens = 200 if output_length == 'Paragraph' else 450
    
    messages = [
        {"role": "system", "content": "You are a skilled writer."},
        {"role": "user", "content": f"Your task is to write a high-quality {output_length.lower()} that has a strong **{sentiment}** tone. The topic is: \"{prompt}\". Provide only the {output_length.lower()}, with no extra commentary or introduction."},
    ]
    
    prompt_for_model = generation_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    try:
        inputs = generation_tokenizer(prompt_for_model, return_tensors="pt").to(generation_model.device)

        outputs = generation_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            eos_token_id=generation_tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        response_only = generation_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return response_only.strip()

    except Exception as e:
        logging.error(f"Error details during text generation: {e}", exc_info=True)
        raise RuntimeError(f"An error occurred during text generation: {e}")