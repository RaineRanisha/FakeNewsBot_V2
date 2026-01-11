import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake News Detection Bot", layout="centered")
st.title("📰 Fake News Detection Chatbot")
st.write("Paste a news article or snippet below and I will predict whether it is fake or real.")

MODEL_REPO = "RayOfLife/FakeBotV2"

@st.cache_resource
def load_model():
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
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs[0][pred].item()

        max_prob = probs.max().item()
        
        if max_prob < 0.65:
            label = "Uncertain"
        else:
            label = "True News" if pred == 1 else "Fake News"

        st.subheader("Result")
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{round(conf*100,2)}%**")
