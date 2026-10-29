import os
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

# Load environment variables from local .env file
load_dotenv()

st.title("📄 AI Document Assistant (RAG)")

# Securely retrieve API Key from environment or Streamlit Cloud Secrets
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and groq_api_key:
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Upgraded to Llama 3.3 70B for strict, accurate RAG answers
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
        
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based strictly on the provided document context below. Do not guess or hallucinate.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        
        rag_chain = (
            {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    st.success("Document processed successfully!")
    
    # Optional debug view so you and your husband can see the actual extracted PDF text
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