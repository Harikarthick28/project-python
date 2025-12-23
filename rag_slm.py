import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------
# Load Small Language Model (SLM)
# ---------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# ---------------------------------------
# Extract text from PDF
# ---------------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------------------------------
# Split text into chunks
# ---------------------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="Resume RAG (SLM)", layout="wide")
st.title("📄 Resume RAG Chatbot (SLM Only)")
st.caption("Upload bulk resumes and search candidates without using any LLM")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []
    resume_sources = []

    for pdf in uploaded_files:
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            resume_sources.append(pdf.name)

    # ---------------------------------------
    # Create embeddings & FAISS index
    # ---------------------------------------
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("✅ Resumes indexed successfully!")

    # ---------------------------------------
    # Query input
    # ---------------------------------------
    query = st.text_input("🔎 Ask about a candidate (skills, experience, role):")

    if query:
        query_embedding = model.encode([query])
        k = 5

        distances, indices = index.search(
            np.array(query_embedding),
            k
        )
        

        st.subheader("🎯 Matching Resume Sections")

        for i, idx in enumerate(indices[0]):
            st.markdown(f"### {i+1}. 📄 Resume: `{resume_sources[idx]}`")
            st.write(all_chunks[idx])
            st.markdown("---")
