from src.helper import generate_document_hash, load_pdf_data, split_text, embed_texts
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Path to store processed document hashes
HASH_FILE_PATH = 'processed_documents_hashes.json'

# Load existing processed document hashes
def load_processed_hashes():
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, 'r') as f:
            return set(json.load(f))
    return set()

# Save processed document hashes
def save_processed_hashes(hashes):
    with open(HASH_FILE_PATH, 'w') as f:
        json.dump(list(hashes), f)

# Filter out processed documents
def filter_processed_documents(documents):
    processed_hashes = load_processed_hashes()
    
    unique_documents = []
    new_hashes = set()
    
    for doc in documents:
        doc_hash = generate_document_hash(doc)
        if doc_hash not in processed_hashes:
            unique_documents.append(doc)
            new_hashes.add(doc_hash)
    
    # Update processed hashes
    processed_hashes.update(new_hashes)
    save_processed_hashes(processed_hashes)
    
    return unique_documents

# Main indexing process
extracted_data = load_pdf_data("Data")
unique_documents = filter_processed_documents(extracted_data)

# Check if there are new documents to process
if not unique_documents:
    print("No new documents to index.")
    exit()

text_chunks = split_text(unique_documents)
embeddings = embed_texts(text_chunks)

# Fetch API Key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set it as an environment variable.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

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
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"Indexed {len(text_chunks)} unique document chunks.")