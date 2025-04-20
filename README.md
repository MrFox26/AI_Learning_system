# 🤖 AI Learning System

An intelligent, personalized learning assistant designed to fetch educational content from diverse sources, personalize learning experiences, and assist with automated report generation using LLMs.

---

## 📦 Setup Instructions

Install required libraries:

```bash
pip install chromadb langchain openai langchain_community requests beautifulsoup4
```

Set up your API Key (if needed):

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

Run the notebook:  
Ensure each cell executes in order, starting from installing dependencies to launching the chatbot with `run_chatbot()`.

---

## 🧠 System Architecture

The system is modular and composed of several core components:

- **Query Generation**  
  Generates user-relevant or topic-specific queries.

- **Data Retrieval**  
  - `bing_results()` – Fetches web content  
  - `wikipedia_summary()` – Retrieves summaries from Wikipedia  
  - `arxiv_papers()` & `semantic_scholar_papers()` – Gathers academic research  
  - `youtube_transcript()` – Extracts transcriptions from educational videos

- **Information Aggregation**  
  - `fetch_information_simulated()` combines results from multiple sources

- **Vector Store**  
  Uses **ChromaDB** to store and manage embeddings for personalized recall.

- **Language Model Interface**  
  Communicates with an LLM (via **LangChain**) to answer questions and generate reports.

- **Chatbot Interface**  
  `run_chatbot()` manages user interactions in a conversational format.

---

## 🧪 Research Methodology

This system simulates an automated research assistant by:

- **Aggregating Multimodal Sources**  
  Text (Wikipedia), academic papers (arXiv/Semantic Scholar), and video transcripts (YouTube)

- **Simulated Query Expansion**  
  Ensures diversity in search using `generate_query()`

- **Structured Knowledge Storage**  
  Saves structured data and summaries in a vector database for semantic retrieval

- **LLM-Enhanced Processing**  
  Uses Groq’s LLM to summarize, explain, and contextualize data for learners

---

## 🎯 Personalization Approach

- **User-Centric Queries**  
  Generated dynamically or provided interactively

- **Vector-Based Context Recall**  
  Stores user-relevant documents and retrieves similar topics using embeddings

- **Multi-Source Fusion**  
  Combines web, academic, and video content to match varied learning styles

- **Interactive Refinement**  
  Chatbot tailors answers and refines outputs based on follow-up prompts

---

## 📝 Report Generation and Modification

- **Report Creation**  
  Generates summaries or topic reports via prompts passed to the LLM using the `ask(prompt)` function

- **LLM-Driven Customization**  
  Users can ask for simplified explanations, keyword-based expansions, or stylistic changes

- **Edit/Refinement Loop**  
  Users interact via `run_chatbot()` to iteratively improve the report quality and relevance

- **Storage and Retrieval**  
  Generated content is optionally stored in **ChromaDB** for future reuse or updates

---

## 🚀 Future Plans

- Add UI for a more visual learning experience  
- Integrate user profiles and learning goals  
- Track learning progress and content effectiveness  
- Support multi-language content processing and delivery

---

Feel free to contribute or fork this project to suit your own learning applications!
