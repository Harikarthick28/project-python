import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("📄 Resume Finder Chatbot (RAG)")
st.write("Upload multiple resume PDFs and search candidates")

# ------------------ CHECK API KEY ------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in environment variables.")
    st.stop()

# ------------------ FILE UPLOAD ------------------
uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    # ------------------ LOAD PDFs ------------------
    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            file_path = tmp.name

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = pdf.name

        documents.extend(docs)

    # ------------------ SPLIT TEXT ------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)

    # ------------------ EMBEDDINGS ------------------
    '''embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )'''
    def load_embeddings():
        return HuggingFaceInferenceAPIEmbeddings(
            api_key="YOUR_HF_TOKEN",   # paste your HuggingFace token here
            model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = load_embeddings()

    # ------------------ VECTOR STORE ------------------
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ------------------ LLM ------------------
    llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)


    # ------------------ PROMPT ------------------
    prompt = PromptTemplate.from_template(
        """
        You are an HR assistant.
        Use the resume context below to answer the question.

        Context:
        {context}

        Question:
        {question}

        If suitable candidates exist, mention their resume file names.
        Answer clearly and professionally.
        """
    )

    # ------------------ RAG CHAIN ------------------
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    st.success("✅ Resumes indexed successfully")

    # ------------------ QUERY ------------------
    query = st.text_input(
        "Ask your question",
        placeholder="Find a Python developer with 3 years experience"
    )

    if st.button("🔍 Search Candidate") and query:
        with st.spinner("Searching resumes..."):
            response = rag_chain.invoke(query)

        st.subheader("🧠 Answer")
        st.write(response)
    

else:
    st.info("Upload resume PDFs to begin")
