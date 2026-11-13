# GenAI IT Helpdesk

An AI-powered IT Helpdesk application built using **Generative AI, Retrieval-Augmented Generation (RAG), LangChain, FAISS, and Ollama**.

The application allows users to ask IT support-related questions and receive intelligent, context-aware responses based on an IT knowledge base.

## Project Overview

Traditional IT helpdesks often require users to search through documentation or wait for technical support.

This project uses a locally running Large Language Model (LLM) combined with a Retrieval-Augmented Generation (RAG) pipeline to provide relevant answers to IT-related queries.

The system retrieves relevant information from stored documents and provides responses using the retrieved context.

## Key Features

- AI-powered IT support chatbot.
- Retrieval-Augmented Generation (RAG) implementation.
- Document processing and text extraction.
- Text chunking using LangChain.
- Semantic search using Hugging Face embeddings.
- FAISS vector database for efficient similarity search.
- Local LLM integration using Ollama.
- Context-based answer generation.
- Chat history management using Streamlit session state.
- Retrieved information display for better transparency.
- Suggested questions for users.
- Fallback responses for questions not supported by the knowledge base.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| LangChain | RAG pipeline orchestration |
| Hugging Face | Text embeddings |
| FAISS | Vector database |
| Ollama | Local LLM execution |
| Llama 3.2 (3B) | Language model |
| PyPDF | PDF text extraction |

## System Architecture

1. **Document Loading:** Load IT support documents and extract their text.
2. **Text Splitting:** Divide documents into smaller chunks using LangChain.
3. **Embedding Generation:** Convert text chunks into numerical vector representations using Hugging Face embeddings.
4. **Vector Storage:** Store embeddings in the FAISS vector database.
5. **User Query:** Accept questions through the Streamlit interface.
6. **Similarity Search:** Retrieve the three most relevant document chunks.
7. **LLM Processing:** Send the retrieved context and user question to the Llama 3.2 model through Ollama.
8. **Response Generation:** Generate a context-based answer and display it to the user.

## Project Structure

```text
GenAI_HelpDesk/
│
├── documents/
│   └── it_support_guide.txt
│
├── vectorstore/
│   └── it_helpdesk_faiss/
│
├── app.py
├── rag_chat.py
├── prepare_documents.py
├── create_vector_database.py
├── test_setup.py
├── test_search.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ShabuirShahin/GenAI_IT_Helpdesk.git
```

### 2. Navigate to the Project

```bash
cd GenAI_IT_Helpdesk
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install and Configure Ollama

Download Ollama from:

https://ollama.com/

Pull the required model:

```bash
ollama pull llama3.2:3b
```

Verify the installed model:

```bash
ollama list
```

### 7. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## How RAG Works in This Project

RAG combines information retrieval with language generation.

Instead of relying only on the language model's pretrained knowledge, the application retrieves relevant information from the IT support knowledge base before generating an answer.

**Workflow:**

User Question → Embedding Generation → FAISS Similarity Search → Relevant Context Retrieval → Llama 3.2 → Generated Answer

## Future Enhancements

- Expand the IT support knowledge base.
- Add more document formats.
- Improve retrieval accuracy.
- Introduce user authentication.
- Add conversation export functionality.
- Deploy the application for wider accessibility.

## Author

**Shabuir Shahin.A**

GitHub: https://github.com/ShabuirShahin

---

*Developed as a Generative AI project to explore Retrieval-Augmented Generation, vector databases, semantic search, and local Large Language Models.*
