
from datetime import datetime
import hashlib
from io import BytesIO

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="GenAI IT Helpdesk",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------
# CUSTOM STYLING
# -------------------------------



st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f4f7fb;
    color: #172b4d !important;
}

/* Main content */
section[data-testid="stMain"] {
    background-color: #f4f7fb;
    color: #172b4d !important;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #16365c !important;
}

/* Main paragraphs and markdown */
[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label,
[data-testid="stMain"] span {
    color: #263b53;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #10243e;
}

section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #344761;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #dce5ef;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Chat message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #263b53 !important;
}

/* Chat input container */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
}

/* Chat input text */
[data-testid="stChatInput"] textarea {
    color: #172b4d !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #172b4d !important;
}

/* Chat input placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1;
}

/* Buttons */
.stButton button {
    background-color: #1769aa;
    color: #ffffff !important;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #12558b;
    color: #ffffff !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #dce5ef;
    border-radius: 10px;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: #263b53 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #7890ad;
    border-radius: 12px;
    padding: 12px;
}

/* Divider */
hr {
    border-color: #e2e8f0;
}

/* Dashboard status cards */
.dashboard-card {
    background: #ffffff;
    border: 1px solid #dce5ef;
    border-radius: 14px;
    padding: 20px;
    min-height: 135px;
    box-shadow: 0 2px 8px rgba(20, 50, 80, 0.04);
}

.dashboard-card h4 {
    color: #64748b !important;
    font-size: 14px;
    margin: 8px 0;
}

.dashboard-card h2 {
    color: #17365c !important;
    font-size: 23px;
    margin: 0;
}

.dashboard-card p {
    color: #64748b !important;
    font-size: 12px;
    margin-top: 8px;
}

.dashboard-heading {
    color: #17365c;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 15px;
}


/* Welcome screen */
.welcome-container {
    background: linear-gradient(135deg, #ffffff, #edf4fc);
    border: 1px solid #dce5ef;
    border-radius: 18px;
    padding: 35px 25px;
    text-align: center;
    margin: 20px 0;
}

.welcome-icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.welcome-title {
    color: #17365c !important;
    font-size: 30px;
    font-weight: 750;
    margin-bottom: 10px;
}

.welcome-description {
    color: #64748b !important;
    font-size: 15px;
}

.section-heading {
    color: #17365c !important;
    font-size: 19px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

div[data-testid="stButton"] button {
    min-height: 48px;
    border-radius: 10px;
    background: #ffffff;
    color: #17365c !important;
    border: 1px solid #dce5ef;
}

div[data-testid="stButton"] button:hover {
    background: #eaf2fc;
    border-color: #3984d6;
    color: #1769aa !important;
}

/* Chat interface improvements */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #dce5ef;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(20, 50, 80, 0.04);
}

/* Chat message text */
[data-testid="stChatMessage"] p {
    color: #263b53 !important;
    line-height: 1.7;
    font-size: 15px;
}

/* Chat timestamps */
.chat-timestamp {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 8px;
}

/* Source information */
.source-card {
    background: #f1f6fc;
    border: 1px solid #dce5ef;
    border-radius: 9px;
    padding: 12px;
    margin: 8px 0;
    color: #263b53;
}

/* Document upload heading */
.upload-heading {
    color: #ffffff;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Document information */
.document-info {
    background: rgba(255,255,255,0.08);
    border: 1px solid #425875;
    border-radius: 9px;
    padding: 10px;
    margin: 7px 0;
    color: #ffffff;
    font-size: 13px;
}

/* Fix sidebar Clear Chat button */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background-color: #1769aa !important;
    color: #ffffff !important;
    border: 1px solid #1769aa !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background-color: #12558b !important;
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------
# LOAD RAG COMPONENTS
# -------------------------------

@st.cache_resource
def load_rag_system():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        "vectorstore/it_helpdesk_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return vector_db, llm, embeddings


# -------------------------------
# APPLICATION HEADER
# -------------------------------

st.title("🤖 GenAI IT Helpdesk")

st.caption(
    "Your AI-powered IT support assistant "
    "using Retrieval-Augmented Generation (RAG)."
)

st.divider()


# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:

    st.title("🛠️ Helpdesk")

    st.markdown("""
    ### About

    This application uses:

    - Python
    - LangChain
    - Hugging Face Embeddings
    - FAISS Vector Database
    - Llama 3.2 LLM
    - Streamlit
    """)

    st.divider()

    st.subheader("Suggested Questions")

    st.markdown("""
    - How do I reset my password?
    - Why is my account locked?
    - How can I get IT support?
    """)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -------------------------------
# INITIALIZE CHAT HISTORY
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# LOAD THE RAG SYSTEM
# -------------------------------

try:

    with st.spinner("Loading AI Helpdesk..."):

        vector_db, llm, embeddings = load_rag_system()

    st.success("AI Helpdesk is ready!")

except Exception as e:

    st.error(f"Unable to load AI system: {e}")
    st.stop()


# -------------------------------
# PROFESSIONAL DASHBOARD
# -------------------------------

st.markdown(
    '<div class="dashboard-heading">System Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

dashboard_cards = [
    (col1, "🗄️", "Vector Database", "FAISS", "Connected"),
    (col2, "🤖", "Language Model", "Llama 3.2", "Running locally"),
    (col3, "🧠", "Embedding Model", "MiniLM", "Semantic search"),
    (col4, "📄", "Documents", "PDF + TXT", "Upload supported")
]

for column, icon, heading, value, description in dashboard_cards:

    with column:

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div style="font-size:25px;">{icon}</div>
                <h4>{heading}</h4>
                <h2>{value}</h2>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")
st.divider()


# -------------------------------
# ADVANCED DOCUMENT MANAGEMENT
# -------------------------------

# Initialize document storage
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = {}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# Rebuild uploaded document vector database
def rebuild_uploaded_vector_db():

    all_chunks = []

    for document_data in st.session_state.uploaded_documents.values():
        all_chunks.extend(document_data["chunks"])

    if all_chunks:

        st.session_state.uploaded_vector_db = FAISS.from_documents(
            all_chunks,
            embeddings
        )

    else:

        st.session_state.pop(
            "uploaded_vector_db",
            None
        )


# Sidebar document manager
st.sidebar.divider()

st.sidebar.markdown(
    '<div class="upload-heading">📁 Document Manager</div>',
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Upload company IT documents to expand the AI knowledge base."
)


# Upload documents
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=f"document_uploader_{st.session_state.uploader_key}"
)


# Process newly uploaded documents
if uploaded_files:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    added_count = 0

    try:

        for file in uploaded_files:

            file_bytes = file.getvalue()

            file_id = hashlib.sha256(
                file.name.encode("utf-8") + file_bytes
            ).hexdigest()

            # Avoid processing duplicate uploads
            if file_id in st.session_state.uploaded_documents:
                continue

            extracted_documents = []

            # PDF extraction
            if file.name.lower().endswith(".pdf"):

                pdf_reader = PdfReader(
                    BytesIO(file_bytes)
                )

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):

                    content = page.extract_text() or ""

                    if content.strip():

                        extracted_documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": file.name,
                                    "page": page_number
                                }
                            )
                        )

            # TXT extraction
            elif file.name.lower().endswith(".txt"):

                content = file_bytes.decode(
                    "utf-8",
                    errors="replace"
                )

                if content.strip():

                    extracted_documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file.name
                            }
                        )
                    )

            # Split into chunks
            chunks = splitter.split_documents(
                extracted_documents
            )

            if chunks:

                st.session_state.uploaded_documents[file_id] = {
                    "name": file.name,
                    "size": len(file_bytes),
                    "chunks": chunks
                }

                added_count += 1

        # Rebuild FAISS index
        if added_count:

            rebuild_uploaded_vector_db()

            st.session_state.upload_notice = (
                f"Successfully processed {added_count} document(s)!"
            )

            # Reset uploader after successful processing
            st.session_state.uploader_key += 1

            st.rerun()

        else:

            st.sidebar.info(
                "No new readable documents were found."
            )

    except Exception as e:

        st.sidebar.error(
            f"Document processing failed: {e}"
        )


# Display upload notification
if st.session_state.get("upload_notice"):

    st.sidebar.success(
        st.session_state.upload_notice
    )

    st.session_state.pop(
        "upload_notice",
        None
    )


# Document statistics
document_count = len(
    st.session_state.uploaded_documents
)

total_chunks = sum(
    len(data["chunks"])
    for data in st.session_state.uploaded_documents.values()
)

total_size = sum(
    data["size"]
    for data in st.session_state.uploaded_documents.values()
)

st.sidebar.divider()

st.sidebar.markdown("### 📊 Knowledge Base")

col_a, col_b = st.sidebar.columns(2)

col_a.metric(
    "Documents",
    document_count
)

col_b.metric(
    "Chunks",
    total_chunks
)

st.sidebar.caption(
    f"Total size: {total_size / 1024:.1f} KB"
)


# Display individual documents
st.sidebar.markdown("### 📂 Uploaded Documents")

if document_count:

    for file_id, data in list(
        st.session_state.uploaded_documents.items()
    ):

        st.sidebar.markdown(
            f"""
            <div class="document-info">
                📄 <b>{data['name']}</b><br>
                <small>
                    {data['size'] / 1024:.1f} KB
                    · {len(data['chunks'])} chunks
                </small><br>
                <small>✓ Processed successfully</small>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.sidebar.button(
            f"🗑️ Remove {data['name']}",
            key=f"remove_{file_id}",
            use_container_width=True
        ):

            st.session_state.uploaded_documents.pop(
                file_id
            )

            rebuild_uploaded_vector_db()

            st.session_state.uploader_key += 1

            st.rerun()


    # Clear all uploaded documents
    if st.sidebar.button(
        "🗑️ Clear All Documents",
        key="clear_all_documents",
        use_container_width=True
    ):

        st.session_state.uploaded_documents = {}

        st.session_state.pop(
            "uploaded_vector_db",
            None
        )

        st.session_state.uploader_key += 1

        st.rerun()

else:

    st.sidebar.info(
        "No uploaded documents yet. Upload a PDF or TXT file to get started."
    )


# -------------------------------
# PROFESSIONAL WELCOME SCREEN
# -------------------------------

if not st.session_state.messages:

    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🤖</div>

        <div class="welcome-title">
            How can we help you today?
        </div>

        <div class="welcome-description">
            Your intelligent IT support assistant is ready.
            Ask a question and get instant troubleshooting guidance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Popular Support Topics</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔑  Password Reset",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "How do I reset my password?"
            )
            st.rerun()

        if st.button(
            "🌐  Network Issues",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "How can I troubleshoot network connectivity issues?"
            )
            st.rerun()

    with col2:

        if st.button(
            "🔒  Account Locked",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "Why is my account locked and how can I unlock it?"
            )
            st.rerun()

        if st.button(
            "💻  Technical Support",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "How can I get technical IT support?"
            )
            st.rerun()

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------


# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        timestamp = message.get("timestamp")

        if timestamp:
            st.caption(f"🕒 {timestamp}")

        # Display source references
        if message.get("sources"):

            with st.expander("📚 View source references"):

                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    # Support old chat history format
                    if isinstance(source, str):
                        st.markdown(
                            f"**Source {index}**"
                        )
                        st.write(source)

                    else:
                        st.markdown(
                            f"**Source {index}: {source['source']}**"
                        )

                        if source.get("page"):
                            st.caption(
                                f"PDF Page: {source['page']}"
                            )

                        st.write(source["content"])

                    st.divider()



# -------------------------------
# USER INPUT
# -------------------------------

typed_question = st.chat_input(
    "Ask your IT support question here..."
)

question = typed_question or st.session_state.pop(
    "pending_question",
    None
)


# -------------------------------
# RAG QUESTION ANSWERING
# -------------------------------

if question:

    st.session_state.messages.append({
    "role": "user",
    "content": question,
    "timestamp": datetime.now().strftime("%I:%M %p")
})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching knowledge base and generating answer..."
        ):

            try:

                # Search original IT knowledge base
                documents = vector_db.similarity_search(
                    question,
                    k=3
                )

                # Search uploaded documents
                if "uploaded_vector_db" in st.session_state:

                    uploaded_results = (
                        st.session_state.uploaded_vector_db
                        .similarity_search(question, k=3)
                    )

                    documents.extend(uploaded_results)

                # Prepare context
                context = "\n\n".join(
                    doc.page_content
                    for doc in documents
                )

                # Prepare RAG prompt
                prompt = f"""
You are a professional IT Helpdesk Assistant.

Answer employee IT support questions using only
the provided knowledge base context.

RULES:

1. Read all retrieved information carefully.
2. If the context directly answers the question,
   provide a clear and helpful answer.
3. If the context contains related information,
   provide only supported troubleshooting steps.
4. Do not invent company policies or procedures.
5. If the exact issue is not covered, clearly
   mention the limitation.
6. If the context contains no useful information,
   respond exactly:
   "I could not find this information in the IT
   support knowledge base."

Keep answers simple and step-by-step.

Knowledge Base Context:
{context}

User Question:
{question}

Answer:
"""

                # Generate answer using Ollama
                response = llm.invoke(prompt)

                answer = response.content

                st.markdown(answer)

                # Display retrieved information
                with st.expander("View retrieved information"):

                    for doc in documents:

                        st.write(
                            f"Source: {doc.metadata.get('source', 'Original Knowledge Base')}"
                        )

                        st.write(doc.page_content)

                        st.divider()

                # Save conversation
                st.session_state.messages.append({
                 "role": "assistant",
                 "content": answer,
                 "timestamp": datetime.now().strftime("%I:%M %p"),
                 "sources": [
                 {
                   "content": doc.page_content,
                   "source": doc.metadata.get(
                   "source",
                   "Original IT Knowledge Base"
                   ),
                   "page": doc.metadata.get("page")
                 }
                 for doc in documents
                 ]
                })

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )
