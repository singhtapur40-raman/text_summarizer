📝 Text Summarization Web Application

A web-based Dialogue Text Summarization Application that leverages a fine-tuned T5-Small Transformer model from Hugging Face to generate concise and meaningful summaries from lengthy conversations. The application combines Natural Language Processing (NLP) with a user-friendly web interface, allowing users to summarize dialogues quickly and efficiently.

📌 Project Overview

This project focuses on abstractive dialogue summarization, where the model understands the context of a conversation and generates a concise summary rather than simply extracting sentences from the input.

The application is built using FastAPI as the backend, HTML, CSS, and JavaScript for the frontend, and a fine-tuned T5-Small model trained on the SAMSum Dialogue Summarization Dataset.

🚀 Features
  Summarizes multi-speaker conversations into concise summaries.
  Fine-tuned Transformer model for dialogue summarization.
  Simple and responsive web interface.
  FastAPI backend for handling inference.
  Handles long conversational inputs.
  Easy to deploy and extend.
  
🛠️ Tech Stack
  Machine Learning
  Python
  Hugging Face Transformers
  T5-Small
  PyTorch
  NumPy
  Pandas
  Scikit-learn
  
Backend
  FastAPI
  Uvicorn
  
Frontend
  HTML
  CSS
  JavaScript

Dataset
  SAMSum Dialogue Summarization Dataset

📊 Future Improvements
  Deploy the application on Hugging Face Spaces or Render.
  Improve summary quality using larger Transformer models such as FLAN-T5 or BART.
  Add support for PDF and document summarization.
  Integrate multilingual summarization.
  Improve UI/UX with loading animations and theme customization.
  Add ROUGE score evaluation.
