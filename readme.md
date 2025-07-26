🚀 [Live API on Render](https://friends-word2vec-api.onrender.com)

# 📺 Friends Character Trait API (Word2Vec NLP Project)

A fun and smart NLP API that captures relationships, traits, and quirks of Friends TV Show characters using Word2Vec and FastAPI.

## 🚀 Features

- `/similar?word=joey` → Get most similar characters/traits
- `/similarity?word1=joey&word2=sandwich` → Check cosine similarity
- `/traits?name=monica` → Get top personality traits for each character

## 🧠 Tech Stack

- Python, FastAPI, Uvicorn
- Gensim Word2Vec
- NLTK preprocessing
- Render.com deployment (public API)

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
