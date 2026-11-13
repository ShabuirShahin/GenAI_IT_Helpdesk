
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# 1. Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load FAISS database
print("Loading FAISS database...")

vector_db = FAISS.load_local(
    "vectorstore/it_helpdesk_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Connect to local LLM
print("Connecting to Llama 3.2...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("\nAI Helpdesk RAG system is ready!")

# 4. Chat loop
while True:

    question = input("\nYou: ")

    if question.lower().strip() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not question.strip():
        continue

    # 5. Retrieve relevant document chunks
    documents = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # 6. Create prompt
    prompt = f"""
You are an IT Helpdesk Assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer is not available, say:
"I could not find this information
in the IT support knowledge base."

Context:
{context}

User Question:
{question}

Answer:
"""

    # 7. Generate answer using LLM
    print("\nAI Helpdesk: Generating answer...")

    response = llm.invoke(prompt)

    print("\nAI Helpdesk:", response.content)
