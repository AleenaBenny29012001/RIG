import streamlit as st
import os
import google.generativeai as genai
from langchain.schema import Document
from typing import List
from pypdf import PdfReader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure API key for Gemini model
os.environ['GOOGLE_API_KEY'] = "AIzaSyBWzzIZOprRuCQV8UKbS2KO15eodKXI05A"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Custom Embeddings using Sentence Transformers
class CustomEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_tensor=True).detach().numpy()

# Initialize custom embeddings instance
embeddings = CustomEmbeddings()

# Function to query the Gemini model
def gemini_model_query(query: str) -> str:
    prompt = f"""You are an assistant for question-answering tasks. Here is the query: '{query}'. 
    If you don't know the answer, just say you don't know."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "".join([page.extract_text() for page in reader.pages])

# Function to split extracted text into chunks
def split_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Function to store documents in FAISS
class FAISSDocumentStore:
    def __init__(self, index_path: str = 'faiss_index'):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.index_path = index_path

        # Create the directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        texts = [doc.page_content for doc in docs]
        embedding_vectors = embeddings.embed_documents(texts)

        print("Generated Embeddings:")
        print(embedding_vectors)

        if self.index is None:
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(embedding_vectors.shape[1])
            self.index.add(embedding_vectors)
        else:
            self.index.add(embedding_vectors)

        # Save the index to disk after adding documents
        self.save_index()
    def save_index(self):
        """Save the FAISS index to the specified directory."""
        faiss.write_index(self.index, os.path.join(self.index_path, 'faiss_index.bin'))
        print(f"FAISS index saved to {os.path.join(self.index_path, 'faiss_index.bin')}")

    def load_index(self):
        """Load the FAISS index from the specified directory."""
        if os.path.exists(os.path.join(self.index_path, 'faiss_index.bin')):
            self.index = faiss.read_index(os.path.join(self.index_path, 'faiss_index.bin'))
            print("FAISS index loaded successfully.")
        else:
            print("No FAISS index found, please add documents first.")


    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        query_vector = embeddings.embed_documents([query])
        _, indices = self.index.search(query_vector, k)  # Search for k nearest neighbors
        return [self.documents[i] for i in indices[0]]

# Initialize FAISS document store
faiss_store = FAISSDocumentStore()

# Function to retrieve relevant documents from FAISS
def get_relevant_passage(query: str, n_results: int = 3) -> str:
    retriever = faiss_store.similarity_search(query, k=n_results)
    return "\n------\n".join(
        f"Content: {doc.page_content}\nMetadata: {doc.metadata}" for doc in retriever
    )

# Create a RAG prompt with relevant passage
def make_rag_prompt(query: str, relevant_passage: str) -> str:
    prompt = f"""
    You are a helpful assistant using the reference passage below to answer the question.
    Keep your response friendly and easy to understand for non-technical users.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")}'
    ANSWER:
    """
    return prompt

# Generate answer using Gemini API
def generate_answer(prompt: str) -> str:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Main function for RAG query
def rag_model_query(query: str) -> str:
    relevant_text = get_relevant_passage(query)
    prompt = make_rag_prompt(query, relevant_text)
    return generate_answer(prompt)

# Streamlit UI setup
st.title("Query Assistant with Gemini, RAG, and RIG Models")

# Sidebar to upload PDF for RAG model
uploaded_file = st.sidebar.file_uploader("Upload a PDF for the RAG model", type="pdf", key="pdf_uploader")

# Input query for all models
query = st.text_input("Enter your query for all models")

# Responses placeholders
gemini_response = ""
rag_response = ""
rig_response = ""

# Process the query if entered
if query:
    # Gemini model response
    gemini_response = gemini_model_query(query)

    # RAG model response if a PDF is uploaded
    if uploaded_file:
        # Extract and store documents in FAISS
        temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load PDF and extract text
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        splitted_docs = split_text(extract_text_from_pdf(temp_file_path))
        
        faiss_store.add_documents(splitted_docs)
        rag_response = rag_model_query(query)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
    else:
        rag_response = "Please upload a PDF to use the RAG model."

    # Placeholder for RIG model response
    rig_response = f"RIG Model Response for Query: '{query}'"

    # Display responses in columns
    st.write("### Responses from All Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Gemini")
        st.write(gemini_response)

    with col2:
        st.header("RAG")
        st.write(rag_response)

    with col3:
        st.header("RIG")
        st.write(rig_response)

else:
    st.write("Please enter a query to get responses from the models.")
