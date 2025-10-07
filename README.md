# 🧠 AI-Powered Sentiment Text Generator

This project is a web application built with **Streamlit** and **Hugging Face Transformers** that analyzes the sentiment of a user-provided prompt and generates a new, original text (a paragraph or an essay) that matches the detected sentiment.  

The entire AI pipeline — from analysis to generation — runs **locally on your machine** without requiring any external APIs.

---

## 📚 Table of Contents
- [How It Works](#how-it-works)
  - [The User Interface (app.py)](#the-user-interface-apppy)
  - [The AI Engine (model.py)](#the-ai-engine-modelpy)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Application](#how-to-run-the-application)
- [Technical Deep Dive](#technical-deep-dive)
  - [Models Used](#models-used)
  - [Core Libraries](#core-libraries)

---

## ⚙️ How It Works

The application is split into two main components:  
a **user-friendly frontend** and a **powerful backend AI engine**.

---

### 🎨 The User Interface (`app.py`)

The user interface is built with **Streamlit** to be simple and interactive. It consists of:

- **Title and Introduction:** Explains what the application does and warns users about the initial model download time.  
- **Input Text Area:** A large box where users can type or paste their text prompt.  
- **"Analyze and Generate" Button:** Starts the analysis and generation process.  
- **Sentiment Display:** Shows whether the AI detected your prompt’s sentiment as *Positive*, *Negative*, or *Neutral*.  
- **Manual Sentiment Override:** A dropdown menu allowing you to change the sentiment manually.  
- **Output Length Selector:** Choose between generating a *Paragraph* or an *Essay*.  
- **Output Display:** Shows the generated text with a loading spinner while the AI processes the request.

---

### 🧩 The AI Engine (`model.py`)

This file handles all AI-related tasks in a two-step process:

#### **Step 1: Sentiment Analysis**
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- **Process:**
  1. The text prompt is sent to `analyze_sentiment_local()`.
  2. The RoBERTa model reads the text and classifies its emotional tone.
  3. Returns a simple label: `positive`, `negative`, or `neutral`.

#### **Step 2: Text Generation**
- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
- **Process:**
  1. The `generate_text_local()` function receives the user’s prompt, detected sentiment, and output length.
  2. Constructs an instruction prompt for TinyLlama, e.g.:
     > "You are a skilled writer. Your task is to write a high-quality paragraph that has a strong positive tone. The topic is: 'The joy of a festival like Diwali with family'."
  3. The model generates a new text following the given sentiment and tone.
  4. The output is returned to the Streamlit app for display.

---

## 🌟 Features

✅ **100% Local:** No API keys or internet required after first download.  
✅ **Automatic Sentiment Detection:** Intelligent emotional tone analysis.  
✅ **Manual Override:** Manually select desired sentiment.  
✅ **Adjustable Length:** Generate concise paragraphs or detailed essays.  
✅ **Interactive UI:** Clean, modern interface powered by Streamlit.

---

## 🧰 Setup and Installation

### **Prerequisites**
- Python 3.8 or newer  
- `pip` (Python package installer)

### **Instructions**

1. **Clone the repository** or download the files:
   ```bash
   git clone https://github.com/your-username/ai-sentiment-generator.git
   cd ai-sentiment-generator

   ---

   ## 🔍 Technical Deep Dive

### 🧠 Models Used

#### **1. Sentiment Analysis**
**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
This **RoBERTa-based model** is fine-tuned on a massive dataset of tweets, making it exceptionally good at understanding the **nuances of sentiment** in modern, informal text.

#### **2. Text Generation**
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
A **compact version of the Llama 2 architecture**, instruction-tuned for chat and text generation tasks.  
It’s ideal for cases where we need to give **structured instructions** (e.g., *“write a positive paragraph about…”*).  
Its **stability**, **instruction-following accuracy**, and **efficient resource usage** make it a great choice for a **local AI application**.

---

### 🧾 Core Libraries

#### **Streamlit**
A Python framework for building and sharing **beautiful, interactive web apps** for machine learning and data science.

#### **Hugging Face Transformers**
The leading library for accessing and using **state-of-the-art pre-trained NLP models**.  
It provides the `pipeline` and `AutoModel` classes used to load and run both sentiment analysis and text generation models.

#### **PyTorch**
A **powerful deep learning framework** that serves as the backend for Hugging Face models.  
The settings:
```python
torch_dtype="auto"
device_map="auto"

