
import streamlit as st
import openai
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


EMBEDDINGS_DIR = "embeddings_folder"
openai.api_key = st.secrets["OPENAI_API_KEY"]


@st.cache_data
def load_all_embeddings(directory):
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_chunks.extend(data)
                except json.JSONDecodeError:
                    continue
    return all_chunks

all_chunks = load_all_embeddings(EMBEDDINGS_DIR)


def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]


def search_chunks(question, top_k=15):
    question_emb = get_embedding(question)
    chunk_embeddings = [chunk["embedding"] for chunk in all_chunks if "embedding" in chunk]
    similarities = cosine_similarity([question_emb], chunk_embeddings)[0]
    ranked = sorted(zip(similarities, all_chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [entry[1]["text"] for entry in ranked[:top_k]]
    return "\n\n".join(top_chunks)


def detect_tone(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.5:
        return "positive"
    elif score <= -0.5:
        return "negative"
    else:
        return "neutral"

def generate_response(question, context, tone):
    system_msg = (
        "You are Tammy, an empathetic and strategic AI mentor for business. "
        "You help users gain clarity, confidence, and momentum."
    )

    tone_prefix = {
        "positive": "Great energy! Let's build on that ",
        "negative": "Thanks for trusting me. I'm here with you. ",
        "neutral": "Let's tackle this together step by step."
    }

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"{tone_prefix[tone]}\n\nQuestion: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.4
    )
    return response.choices[0].message["content"]


st.title(" Tammy – Your Business Mentor")
st.markdown("Welcome to your strategic clarity hub. Ask Tammy anything about business, mindset, or growth.")

question = st.text_input(" Enter your question for Tammy:")

if st.button("Ask Tammy"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            tone = detect_tone(question)
            context = search_chunks(question)
            answer = generate_response(question, context, tone)
        st.success(" Tammy's Response:")
        st.write(answer)
