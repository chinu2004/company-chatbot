# 🧑‍💻 Mr. Chinu — Codebasics FAQ Chatbot

An AI-powered FAQ chatbot that answers questions about **Codebasics** tutorials and courses — built with Mistral AI, LangChain, FAISS, and Streamlit.

## Live Demo:https://company-chatbot-attppywfhupjsipeyfgwc4.streamlit.app/
---

## ✨ Features

- 💬 **Natural language Q&A** — ask anything about Codebasics content
- 🔍 **Semantic search** with FAISS vector store — finds the most relevant FAQ entries
- 🧠 **Mistral Large** LLM generates clean, context-grounded answers
- 🚫 **Out-of-scope handling** — politely refuses unrelated questions
- ⚡ **Fast retrieval** using pre-built FAISS index (no rebuild on every run)
- 🖥️ **Clean Streamlit UI** with a spinner and formatted response

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | [Mistral Large](https://mistral.ai/) via `langchain-mistralai` |
| Embeddings | `intfloat/e5-base` via HuggingFace |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Frontend | [Streamlit](https://streamlit.io/) |
| Data | CSV (`codebasics_faqs.csv`) |

---

## 📁 Project Structure

```
company-chatbot/
│
├── main.py                 # Streamlit app — loads FAISS & runs Q&A
├── build_faiss.py          # One-time script to build the FAISS index
├── codebasics_faqs.csv     # FAQ dataset (prompt + response columns)
├── faiss_index/            # Pre-built FAISS vector index
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/chinu2004/company-chatbot.git
cd company-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.streamlit/secrets.toml` file:

```toml
MISTRAL_API_KEY = "your_mistral_api_key_here"
```

Get your API key at [console.mistral.ai](https://console.mistral.ai).

### 5. Build the FAISS index (first time only)

```bash
python build_faiss.py
```

This reads `codebasics_faqs.csv`, generates embeddings, and saves the index to `faiss_index/`. The pre-built index is already committed to the repo, so skip this unless you update the CSV.

### 6. Run the app

```bash
streamlit run main.py
```

---

## 📋 How It Works

```
User Question
     │
     ▼
HuggingFace e5-base Embeddings
     │
     ▼
FAISS Retriever (top 5 matches)
     │
     ▼
PromptTemplate (context + question)
     │
     ▼
Mistral Large LLM
     │
     ▼
Answer (grounded in FAQ context only)
```

1. The user's question is embedded using `intfloat/e5-base`.
2. FAISS retrieves the top 5 most semantically similar FAQ entries.
3. A strict prompt instructs the LLM to answer **only** from the retrieved context.
4. If no relevant context is found, the bot responds with a graceful fallback.

---

## 📄 FAQ Data Format

The `codebasics_faqs.csv` must have these two columns:

```csv
prompt,response
"What courses does Codebasics offer?","Codebasics offers courses on..."
"How do I get a certificate?","After completing the course..."
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-mistralai
faiss-cpu
sentence-transformers
```

> See `requirements.txt` for the full list.

---

## 📄 License

MIT License. Free to use and adapt for your own company's FAQ dataset.
