import re
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
st.set_page_config(page_title="Fake News Detection Bot", layout="centered")
st.title("📰 Fake News Detection Chatbot")
st.write("Paste a news article or snippet below and I will predict whether it is fake or real.")
 
# FIXED: training-time preprocessing must be replicated at inference time
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "this", "that", "these", "those", "it", "its",
}
 
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)
    text = text.lower()
    return " ".join(w for w in text.split() if w not in STOP_WORDS)
 
 
@st.cache_resource
def load_model():
    MODEL_REPO = "RayOfLife/FakeBotV2"
    # FIXED: these two lines were outside load_model() (no indentation),
    # so MODEL_REPO was out of scope and the lines below were unreachable
    # due to a broken indentation block (IndentationError on save).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    return tokenizer, model
 
 
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
 
user_input = st.text_area("Enter news text:", height=200)
 
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)  # FIXED: match training preprocessing
        inputs = tokenizer(cleaned_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs[0][pred].item()
 
        label = "True News" if pred == 1 else "Fake News"
        st.subheader("Result")
        st.write(f"🧠 Prediction: **{label}**")
        st.write(f"📊 Confidence: **{round(conf*100,2)}%**")
 
 
!zip -r distilbert-fake-news.zip distilbert-fake-news
from google.colab import files
 
files.download("app.py")
files.download("distilbert-fake-news.zip")  # FIXED: zip was created but never downloaded
 
