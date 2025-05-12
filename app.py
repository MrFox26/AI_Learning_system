# app.py
import streamlit as st
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import requests
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- UI: Collect user preferences ---
st.title("🧠 AI-Powered Learning Assistant")

with st.form("preferences_form"):
    topic = st.text_input("What topic do you want to learn about?")
    objective = st.text_input("What is your learning goal?")
    knowledge = st.selectbox("Your current knowledge level", ["Beginner", "Intermediate", "Advanced"])
    preferred_format = st.multiselect("Preferred format", ["Text", "Videos", "Diagrams", "Examples", "All"], default=["All"])
    submitted = st.form_submit_button("Generate Learning Report")

if submitted and topic:
    with st.spinner("Gathering information..."):

        # --- Convert preferences to query ---
        query = f"I'm a {knowledge.lower()} learner interested in '{topic}'. " \
                f"My goal is to learn about {objective}, and I prefer learning through {', '.join(preferred_format).lower()}."

        # --- Load content sources ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        data1_split, data2_split, data4_split = [], [], []

        if "Text" in preferred_format or "All" in preferred_format:
            try:
                wiki_loader = WikipediaLoader(query=topic, load_max_docs=5)
                wiki_data = wiki_loader.load()
                data1_split = text_splitter.split_documents(wiki_data)
            except:
                st.warning("Wikipedia loading failed.")

            try:
                arxiv_loader = ArxivLoader(query=topic, load_max_docs=5)
                arxiv_data = arxiv_loader.load()
                data2_split = text_splitter.split_documents(arxiv_data)
            except:
                st.warning("ArXiv loading failed.")

        if "Videos" in preferred_format or "All" in preferred_format:
            YT_API_KEY = os.getenv("YOUTUBE_API_KEY")
            if not YT_API_KEY:
                st.error("Missing YouTube API Key. Set YOUTUBE_API_KEY in .env")
            else:
                yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&q={topic}&maxResults=5&key={YT_API_KEY}"
                response = requests.get(yt_url)
                yt_links = [
                    f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                    for item in response.json().get("items", [])
                ]

                for link in yt_links:
                    try:
                        yt_loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)
                        yt_data = yt_loader.load()
                        data4_split.extend(text_splitter.split_documents(yt_data))
                    except Exception as e:
                        st.warning(f"Skipping video due to error: {e}")

        # --- Create vector store and retrieve ---
        all_docs = data1_split + data2_split + data4_split
        if not all_docs:
            st.error("No content could be loaded.")
            st.stop()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = Chroma(
            collection_name="learning_assistant",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db"
        )
        vector_store.add_documents(all_docs)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 30})
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # --- Prompt & LLM ---
        template = """
        You are an AI educational assistant. Based on the provided context and query, generate a detailed, well-structured educational report in **markdown** format with the following **exact** sections:

        1. Title  
        2. Introduction  
        3. Learning Objectives  
        4. Concept Breakdown  
        5. Visual Aids Description  
        6. Examples and Use Cases  
        7. Citations & References  
        8. Recommended Additional Resources  
        9. Conclusion and Next Steps

        Query: {query}

        Educational Content:
        {context}
        """
        prompt = PromptTemplate(input_variables=["query", "context"], template=template)
        final_prompt = prompt.invoke({"query": query, "context": context})

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Missing GROQ_API_KEY in .env file.")
            st.stop()

        os.environ["GROQ_API_KEY"] = groq_api_key
        llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
        response = llm.invoke(final_prompt)

        # --- Display report ---
        st.subheader("📄 Generated Learning Report")
        st.markdown(response.content, unsafe_allow_html=True)
