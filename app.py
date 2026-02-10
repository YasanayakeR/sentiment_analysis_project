from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import string
import pickle
from nltk.stem import PorterStemmer

app = FastAPI()

# Load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    stopwords = file.read().splitlines()

# Load vocabulary
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

ps = PorterStemmer()

# Pydantic model for request
class Review(BaseModel):
    review: str

# Preprocessing functions
def remove_punctuations(text: str) -> str:
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text: str) -> pd.Series:
    data = pd.DataFrame([text], columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r"https?://\S+|www\.\S+", '', x) for x in x.split()))
    data['tweet'] = data['tweet'].apply(remove_punctuations)
    data['tweet'] = data['tweet'].str.replace('\d+', '', regex=True)
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(word for word in x.split() if word not in stopwords))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(ps.stem(word) for word in x.split()))
    return data['tweet']

def vectorizer(ds: pd.Series, vocabulary: list) -> np.ndarray:
    vectorized_lst = []
    for sentence in ds:
        sentence_vec = np.zeros(len(vocabulary))
        for i, word in enumerate(vocabulary):
            if word in sentence.split():
                sentence_vec[i] = 1
        vectorized_lst.append(sentence_vec)
    return np.asarray(vectorized_lst, dtype=np.float32)

# Prediction endpoint
@app.post("/predict")
def predict(review: Review):
    if not review.review:
        raise HTTPException(status_code=400, detail="No review text provided")
    
    try:
        preprocessed_text = preprocessing(review.review)
        vectorized_text = vectorizer(preprocessed_text, tokens)
        prediction = model.predict(vectorized_text)
        sentiment = "Positive" if prediction == 0 else "Negative" if prediction == 1 else "Neutral"
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
