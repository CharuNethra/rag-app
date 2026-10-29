import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.title("📄 AI Document Assistant (RAG)")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Clear old session states if a new file is uploaded
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file
    # Generate a unique collection identifier per upload to avoid caching/stale data
    st.session_state.collection_id = str(uuid.uuid4())

if uploaded_file and groq_api_key:
    with st.spinner("Processing document..."):
        # Save uniquely to prevent collision or stale read
        temp_file_path = f"temp_{st.session_state.collection_id}.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create an isolated, in-memory vector store unique to this specific file upload
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name=f"rag_col_{st.session_state.collection_id}"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Using Groq's active supported model
        llm = ChatGroq(model_name="openai/gpt-oss-20b", groq_api_key=groq_api_key)
        
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based strictly on the provided document context below. Do not guess, assume, or hallucinate.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        
        rag_chain = (
            {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    st.success("Document processed successfully!")
    
    # Debug viewer so you can verify the text belongs to the correct file instantly
    with st.expander("🔍 View Extracted Document Content"):
        for i, doc in enumerate(docs):
            st.write(f"**Page {i+1}:** {doc.page_content[:500]}...")

    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
            st.write("### Answer:")
            st.write(response)

elif uploaded_file and not groq_api_key:
    st.error("API Key not found in environment or secrets settings.")