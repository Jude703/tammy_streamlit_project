
import streamlit as st
import openai
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# ========== إعدادات ==========

# 🔑 مفتاح OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# 📁 مسارات
EMBEDDINGS_DIR = "embeddings_folder"
PERSONA_PATH = "tammy_persona.txt"
USER_MEMORY_PATH = "user_memory.json"

# ========== تحميل الذاكرة والشخصية ==========
def load_persona(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_user_memory(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_memory(memory, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

tammy_persona = load_persona(PERSONA_PATH)
user_memory = load_user_memory(USER_MEMORY_PATH)

# ========== تحميل كل الملفات من مجلد embeddings ==========
def load_all_embeddings(embeddings_dir):
    all_chunks = []
    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".json"):
            path = os.path.join(embeddings_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_chunks.extend(data)
    return all_chunks

all_chunks = load_all_embeddings(EMBEDDINGS_DIR)

# ========== تحليل المشاعر ==========
sentiment_analyzer = SentimentIntensityAnalyzer()

def detect_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.5:
        return "positive"
    elif compound <= -0.5:
        return "negative"
    else:
        return "neutral"

# ========== استرجاع أفضل chunks ==========
def retrieve_relevant_chunks(user_query, chunks, top_k=5):
    response = openai.Embedding.create(input=[user_query], model="text-embedding-ada-002")
    query_embedding = np.array(response["data"][0]["embedding"])

    chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

    scored_chunks = sorted(zip(similarities, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

# ========== إنشاء الرد النهائي ==========
def generate_response(query, retrieved_chunks, sentiment, memory):
    context = "\n\n".join([f"[{chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks])
    past = "\n\n".join([f"User: {item['user']}\nTammy: {item['tammy']}" for item in memory[-5:]])

    prompt = f"""{tammy_persona}

Sentiment: {sentiment}
User message: {query}

Previous Memory:
{past}

Relevant Context:
{context}

Answer Tammy-style:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()

# ========== واجهة Streamlit ==========

st.set_page_config(page_title="Ask Tammy", page_icon="🪄", layout="centered")

st.title("✨ Ask Tammy — Your Emotional & Strategic AI Mentor")
st.markdown("Welcome! I'm **Tammy**, your AI mentor and clarity cofounder. Ask me anything.")

user_input = st.text_input("💬 What's on your mind?")

if st.button("Ask Tammy") and user_input:
    with st.spinner("Thinking with empathy..."):
        sentiment = detect_sentiment(user_input)
        top_chunks = retrieve_relevant_chunks(user_input, all_chunks, top_k=8)
        answer = generate_response(user_input, top_chunks, sentiment, user_memory)

        st.markdown("#### 🪄 Tammy's Answer")
        st.write(answer)

        user_memory.append({"user": user_input, "tammy": answer})
        save_user_memory(user_memory, USER_MEMORY_PATH)

if st.button("🗑️ Clear Memory"):
    user_memory = []
    save_user_memory(user_memory, USER_MEMORY_PATH)
    st.success("Memory cleared successfully.")
