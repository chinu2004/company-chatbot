from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load CSV
loader = CSVLoader(
    file_path="codebasics_faqs.csv",
    source_column="prompt",
    encoding="latin-1"
)

docs = loader.load()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base",
    encode_kwargs={"normalize_embeddings": True}
)

# Create FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)

# Save index
vectorstore.save_local("faiss_index")

print("✅ FAISS index created successfully")
