import os
import tiktoken
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv




HUMAN_TEMPLATE = """

You are a helpful assistant who answers questions based on provided context. 
You must only use the provided context, and cannot use your own knowledge. 
If you do not know the answer, or it's not contained in the provided context response with "I don't know"

#Question:
{query}

#CONTEXT:
{context}
"""

def check_api_key():
    load_dotenv()
    """Verify that the API key is set and valid"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return api_key

#Read PDF data
def read_pdf_data(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        if not text.strip():
            raise ValueError("No text extracted from PDF")
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def tiktoken_len(text):
    try:
        tokens = tiktoken.encoding_for_model("gpt-4").encode(text)
        return len(tokens)
    except Exception as e:
        raise Exception(f"Error in token calculation: {str(e)}")

#Split data into chunks
def split_data(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200,  # Added overlap for better continuity
        )   
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise ValueError("Text splitting produced no chunks")
        return chunks
    except Exception as e:
        raise Exception(f"Error splitting text: {str(e)}")

#Create embeddings instance

def create_embeddings():
    try:
        api_key = check_api_key()
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        return embedding_model
    except Exception as e:
        raise Exception(f"Error creating embeddings model: {str(e)}")


# Create a vector database using Qdrant
def create_vector_store(embedding_model, chunks):
    try:
        embedding_dim = 1536
        client = QdrantClient(":memory:")  # Consider using persistent storage for production
        
        # Create collection with error handling
        try:
            client.create_collection(
                collection_name="lcel_doc_v2",
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
        except Exception as e:
            raise Exception(f"Error creating Qdrant collection: {str(e)}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="lcel_doc_v2",
            embedding=embedding_model,
        )
        
        # Add texts with progress tracking
        try:
            _ = vector_store.add_texts(texts=chunks)
        except Exception as e:
            raise Exception(f"Error adding texts to vector store: {str(e)}")
            
        return vector_store
    except Exception as e:
        raise Exception(f"Error in vector store creation: {str(e)}")

# create RAG
def create_rag():
    try:
        api_key = check_api_key()
        openai_chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context."),
            ("human", HUMAN_TEMPLATE)
        ])
        if 'vector_store' in st.session_state:
            vector_store = st.session_state.vector_store        
        else:
            raise ValueError("Vector store not found in session state")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        simple_rag = (
            {"context": retriever, "query": RunnablePassthrough()}
            | chat_prompt
            | openai_chat_model
            | StrOutputParser() 
        ) 
        
        return simple_rag
    except Exception as e:
        raise Exception(f"Error creating RAG chain: {str(e)}")

# Invoke RAG
def invoke_rag(query):
    try:
        rag_chain = create_rag()
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        raise Exception(f"Error invoking RAG chain: {str(e)}")




