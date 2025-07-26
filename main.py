import pandas as pd
import difflib

# # Load Excel file
# file_path = r"C:\Users\abhishekk\Downloads\Junk\mapping col A and B.xlsx"
# df = pd.read_excel(file_path)
#
# # Extract columns
# col_a = df["Column A"].dropna().astype(str).tolist()
# col_b = df["Column B"].dropna().astype(str).tolist()
#
# # Perform string similarity matching
# mapped_data = []
# for a in col_a:
#     match = difflib.get_close_matches(a, col_b, n=1, cutoff=0)
#     best_match = match[0] if match else "NA"
#     score = difflib.SequenceMatcher(None, a, best_match).ratio() if best_match != "NA" else 0
#     mapped_data.append((a, best_match, round(score * 100, 2)))
#
# # Save results to a new file (avoid overwriting the original input)
# output_path = r"C:\Users\abhishekk\Downloads\Junk\mapped_output.xlsx"
# mapped_df = pd.DataFrame(mapped_data, columns=["Column A", "Mapped Column B", "Match Score (%)"])
# mapped_df.to_excel(output_path, index=False)
#
# print("Mapping complete. Results saved to 'mapped_output.xlsx'.")

from fastapi import FastAPI, HTTPException
from gensim.models import Word2Vec

app = FastAPI()
# model = Word2Vec.load("friends_word2vec_with_phrases.model")

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