from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 

def load_pdf_data(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# extracted_data = load_pdf_data("Data") 

def split_text(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def ollama_embeddings():
   embeddings = OllamaEmbeddings(model="nomic-embed-text")
   return embeddings
# embedded_texts = embed_chunks(text_chunks, embeddings)

