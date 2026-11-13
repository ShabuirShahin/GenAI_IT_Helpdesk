
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS database
vector_db = FAISS.load_local(
    "vectorstore/it_helpdesk_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

# Search for relevant information
question = "How can I reset my forgotten password?"

results = vector_db.similarity_search(question, k=2)

print("\nQuestion:", question)
print("\nRelevant information:")

for result in results:
    print("\n", result.page_content)
