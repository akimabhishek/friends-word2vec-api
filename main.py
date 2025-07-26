import pandas as pd
import difflib
import os
import requests
import gensim
from fastapi import FastAPI, HTTPException
from gensim.models import Word2Vec

app = FastAPI()

MODEL_PATH = "friends_word2vec_with_phrases.model"
MODEL_URL = "https://huggingface.co/akimabhi/friends-word2vec/resolve/main/friends_word2vec_with_phrases.model"

# Step 1: Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded!")

# Step 2: Load the model
print("📦 Loading Word2Vec model...")
model = gensim.models.Word2Vec.load(MODEL_PATH)
print("✅ Model loaded!")

@app.get("/")
def root():
    return {"message": "Friends Word2Vec API"}

@app.get("/similar")
def similar(word: str, topn: int = 5):
    try:
        similar_words = model.wv.most_similar(word.lower(), topn=topn)
        return {"word": word, "similar": similar_words}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{word}' not in vocabulary")

@app.get("/similarity")
def similarity(word1: str, word2: str):
    try:
        sim = model.wv.similarity(word1.lower(), word2.lower()) * 100
        return {"word1": word1, "word2": word2, "similarity": sim}
    except KeyError:
        raise HTTPException(status_code=404, detail="One or both words not in vocabulary")

@app.get("/traits")
def traits(name: str, topn: int = 5):
    try:
        similar_words = model.wv.most_similar(name.lower(), topn=topn)
        traits = [word for word, _ in similar_words]
        return {"character": name, "traits": traits}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{name}' not in vocabulary")