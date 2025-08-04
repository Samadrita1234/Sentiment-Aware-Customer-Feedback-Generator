# streamlit_app.py

import streamlit as st
import re
import subprocess
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = fetch_20newsgroups(subset='train', categories=['rec.autos', 'sci.med'])
    texts = data.data[:1000]
    labels = [1 if 'good' in text.lower() or 'great' in text.lower() else 0 for text in texts]
    texts_clean = [re.sub(r"[^a-zA-Z0-9 ]", "", t.lower()) for t in texts]
    return texts_clean, labels

# Train sentiment classifier
@st.cache_resource
def train_classifier(texts_clean, labels):
    X_train, X_test, y_train, y_test = train_test_split(texts_clean, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# Generate response from local LLM
def generate_response(prompt):
    try:
        result = subprocess.run(
            ['ollama', 'run', 'mistral'],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout.decode('utf-8')
    except FileNotFoundError:
        return "Error: Ollama or Mistral model not found. Please install from https://ollama.com."

# Streamlit UI
st.title("🧠 AI-Powered Customer Feedback Generator")
st.markdown("This app classifies customer reviews and generates intelligent replies using a free local LLM.")

user_input = st.text_area("✍️ Enter a customer review:", height=150)

if user_input:
    texts_clean, labels = load_data()
    model, vectorizer = train_classifier(texts_clean, labels)

    clean_input = re.sub(r"[^a-zA-Z0-9 ]", "", user_input.lower())
    vec_input = vectorizer.transform([clean_input])
    prediction = model.predict(vec_input)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    prompt = (
        f"Write a polite and cheerful response to this positive customer review: '{user_input}'"
        if prediction == 1 else
        f"Write an empathetic and apologetic response to this negative customer review: '{user_input}'"
    )

    st.markdown(f"**🧾 Detected Sentiment:** `{sentiment}`")
    st.markdown("**📝 Generated Response:**")
    response = generate_response(prompt)
    st.code(response)

    if "Error:" in response:
        st.error(response)
