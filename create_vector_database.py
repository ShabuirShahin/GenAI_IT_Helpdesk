
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Read the IT support document
file_path = Path("documents/it_support_guide.txt")
text = file_path.read_text(encoding="utf-8")

print("Document loaded!")

# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)
print("Total chunks:", len(chunks))

# 3. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# 4. Convert chunks into vectors and store in FAISS
vector_db = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

# 5. Save the vector database
vector_db.save_local("vectorstore/it_helpdesk_faiss")

print("FAISS vector database created successfully!")
