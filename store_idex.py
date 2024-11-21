from src.helper import load_pdf_data,split_text,ollama_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

extracted_data = load_pdf_data("Data") 
text_chunks = split_text(extracted_data)
embeddings = ollama_embeddings()

# Fetch API Key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set it as an environment variable.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot-1"

# Create an index
pc.create_index(
    name=index_name,
    dimension=768,  # Replace with your model's vector dimension
    metric="cosine",  # Similarity metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name=index_name,
    embedding= embeddings,
)

