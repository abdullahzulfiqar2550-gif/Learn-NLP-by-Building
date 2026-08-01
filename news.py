import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import re
import pickle

app = FastAPI()

model1 = pickle.load(open(r"C:\Users\Abdul\OneDrive\Documents\Ultimate AI Mastery BootCamp\NLP\Session 28,29&30\news_vectorizer.pkl",'rb'))
model2 = pickle.load(open(r"C:\Users\Abdul\OneDrive\Documents\Ultimate AI Mastery BootCamp\NLP\Session 28,29&30\news_label_encoder.pkl",'rb'))
model3 = pickle.load(open(r"C:\Users\Abdul\OneDrive\Documents\Ultimate AI Mastery BootCamp\NLP\Session 28,29&30\news_category_model.pkl",'rb'))

@app.get('/')
def index():
    return {'Deployment': 'Hello and Welcome to AI Engineering '}

@app.post('/predict')
def nlp(text : str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    x = model2.transform([text])
    prediction = model1.predict(x)

    output = model3.inverse_transform(prediction)[0]

    return {"Prediction": output}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000s)