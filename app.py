import streamlit as st
import os
import chromadb
import requests
from bs4 import BeautifulSoup
import urllib.parse
import random
import time

st.set_page_config(page_title="Personalized Learning Assistant", layout="wide")

# Query generation
def generate_query(topic, objective):
    return f"How to learn {topic} to achieve {objective}"

# Simulated Bing search results
def bing_results(query):
    return [f"https://example.com/search_result_{i}?q={urllib.parse.quote(query)}" for i in range(1, 6)]

# Wikipedia summary fetching
def wikipedia_summary(query):
    search_url = f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(query)}"
    try:
        response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', class_='mw-parser-output')
        if content:
            for para in content.find_all('p'):
                if para.text.strip():
                    return para.text.strip()
        return "No Wikipedia summary found."
    except requests.RequestException as e:
        return f"Error fetching Wikipedia summary: {str(e)}"

def arxiv_papers(query):
    return [(f"Simulated Paper Title {i}", f"Simulated abstract for {query}", f"https://arxiv.org/abs/1234.{random.randint(1000,9999)}") for i in range(1, 4)]

def semantic_scholar_papers(query):
    return [(f"Simulated Research Title {i}", f"Simulated research abstract on {query}", f"https://semanticscholar.org/paper/{random.randint(100000,999999)}") for i in range(1, 4)]

def youtube_transcript(query):
    return " ".join([
        f"In this video, we discuss {query} in depth.",
        f"The basics of {query} are explained thoroughly.",
        f"We move into advanced techniques.",
        f"Finally, we wrap up the session with key takeaways."
    ])

# Fetch info from simulated sources
def fetch_information_simulated(query):
    with st.spinner("Fetching resources..."):
        info = {
            'Bing_Web_Results': bing_results(query),
            'Wikipedia_Summary': wikipedia_summary(query),
            'Arxiv_Papers': arxiv_papers(query),
            'Semantic_Scholar_Papers': semantic_scholar_papers(query),
            'YouTube_Transcript': youtube_transcript(query)
        }
        time.sleep(1)
        return info

# Store in ChromaDB
def store_to_chroma(info_dict, query, collection_name="learning_resources"):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(collection_name)

    documents, metadatas, ids = [], [], []
    counter = 0
    for source, content in info_dict.items():
        if isinstance(content, list):
            for item in content:
                text = " ".join(item) if isinstance(item, tuple) else str(item)
                documents.append(text)
                metadatas.append({"query": query, "source": source})
                ids.append(f"{source}_{counter}")
                counter += 1
        else:
            documents.append(str(content))
            metadatas.append({"query": query, "source": source})
            ids.append(f"{source}_{counter}")
            counter += 1

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

# Generate structured report using Groq API
def generate_structured_report(query, collection_name="learning_resources", model="llama3-8b-8192"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY in environment variables.")
        return "API key error."

    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query(query_texts=[query], n_results=10)
    relevant_docs = results.get('documents', [[]])[0]
    combined_context = "\n\n".join(relevant_docs) if relevant_docs else "No relevant documents found."

    prompt = f"""
You are a learning assistant. Based on the following educational content, generate a detailed, structured report. Include:

1. Introduction
2. Learning Objectives
3. Core Concepts
4. Visual Aids (Mermaid syntax)
5. Citations
6. Next Steps

Query: {query}

Educational Content:
{combined_context}
"""

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(groq_api_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No text generated.")

# Streamlit UI
st.title("🎓 Personalized Learning Assistant")
with st.sidebar:
    st.header("🧠 Your Preferences")
    topic = st.text_input("What topic are you interested in?")
    level = st.selectbox("How familiar are you?", ["Beginner", "Intermediate", "Expert"])
    format_pref = st.radio("Preferred format", ["Videos", "Articles", "Quizzes", "Projects"])
    objective = st.text_area("What's your learning objective?")
    submit = st.button("Generate Report")

if submit:
    if topic and objective:
        query = generate_query(topic, objective)
        info = fetch_information_simulated(query)
        store_to_chroma(info, query)
        report = generate_structured_report(query)
        st.subheader("📘 Generated Learning Report")
        st.markdown(report)
    else:
        st.warning("Please provide both a topic and objective.")
