import streamlit as st
from transformers import pipeline

# Initialize pipelines (Hugging Face models)
@st.cache_resource
def init_pipelines():
    sentiment_pipeline = pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    response_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")  # Lightweight GPT-2
    return sentiment_pipeline, response_pipeline

sentiment_analyzer, response_generator = init_pipelines()

# Streamlit UI
st.title("🧠 AI Feedback Generator")
st.markdown("Enter customer feedback and get sentiment detection plus a polite AI response!")

user_feedback = st.text_area("✍️ Enter your feedback here:", height=150)

if user_feedback:
    # Sentiment Analysis
    result = sentiment_analyzer(user_feedback)[0]
    sentiment = result['label']
    confidence = result['score']
    
    st.markdown(f"**🧾 Detected Sentiment:** `{sentiment}` (Confidence: {confidence:.2f})")
    
    # Generate AI Response
    if sentiment == "POSITIVE":
        prompt = f"Write a cheerful and polite reply to this positive feedback: '{user_feedback}'"
    elif sentiment == "NEGATIVE":
        prompt = f"Write an empathetic and apologetic reply to this negative feedback: '{user_feedback}'"
    else:
        prompt = f"Write a neutral, polite reply to this feedback: '{user_feedback}'"
    
    # response = response_generator(prompt, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']
    # response = response_generator(prompt, max_length=60, truncation=True)[0]['generated_text']
    # response_generator(prompt, max_length=100, truncation=True)
    response = response_generator(
    prompt,
    max_new_tokens=80,
    do_sample=True,       # enables random sampling
    temperature=0.7,      # creativity
    top_p=0.9,            # nucleus sampling
    repetition_penalty=2.0, # discourages repeated phrases
    truncation=True
)[0]['generated_text']


    st.markdown("**📝 AI Generated Response:**")
    st.write(response)
