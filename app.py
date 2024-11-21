from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from src.helper import ollama_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompts import *
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to make requests

# Load API keys
groq_api_key = os.environ.get('GROQ_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize embeddings and vector store
embeddings = ollama_embeddings()
index_name = "medical-chatbot-1"

# Create vector store and retriever
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=500,
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create chains
question_answering_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

# Route to serve the chat interface
@app.route('/')
def home():
    return render_template('medical_chat.html')

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get message from request
        data = request.json
        msg = data.get('message', '')
        
        # Invoke RAG chain
        response = rag_chain.invoke({"input": msg})
        
        # Return bot's answer
        return jsonify({
            "response": response["answer"]
        })
    
    except Exception as e:
        # Error handling
        print(f"Error processing message: {e}")
        return jsonify({
            "response": "Sorry, I encountered an error processing your request."
        }), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)