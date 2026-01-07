import streamlit as st
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate

# 🔐 API key from Streamlit Secrets
api_key = st.secrets["MISTRAL_API_KEY"]

llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=api_key,
    temperature=0.6,
    max_tokens=1000
)

st.title("Mr. Chinu 🧑‍💻")
st.write("Feel free to ask any query related to the Codebasics tutorials! 💬")

main_placeholder = st.empty()

# 📄 Load CSV from repo
loader = CSVLoader(
    file_path="codebasics_faqs.csv",
    source_column="prompt"
)
data = loader.load()

# 🔎 Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base",
    encode_kwargs={"normalize_embeddings": True}
)

# 📦 Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Question Answering bot.

Each context contains a question and its answer in this format:
prompt: <question>
response: <answer>

Rules:
- Answer ONLY using the response part from the context.
- If the question is not related to the context, say:
  "I am sorry, I am not able to help with that."

Context:
{context}

User Question:
{question}

Final Answer:
"""
)

def qa_bot(query):
    docs = retriever.invoke(query)

    if not docs:
        return "I am sorry, I am not able to help with that."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = QA_PROMPT.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content

query = main_placeholder.text_input("Type your question here...")
if query:
    with st.spinner("Thinking..."):
        answer = qa_bot(query)
        time.sleep(1)

    st.subheader("😇 Response:")
    st.write(answer)
