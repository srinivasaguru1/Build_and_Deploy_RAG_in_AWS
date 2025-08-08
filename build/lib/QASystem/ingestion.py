from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
import boto3
import os

# Setup Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Ingest and split documents
def data_ingestion():
    loader = PyPDFDirectoryLoader("../data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Loaded {len(docs)} chunks.")
    return docs

# Create FAISS vector store
def get_vector_store(docs):
    if not docs:
        raise ValueError("No documents to index.")
    
    # Test embedding to check if it's working
    try:
        test = bedrock_embeddings.embed_query("test")
        print(f"Bedrock embedding returned vector of length: {len(test)}")
    except Exception as e:
        raise RuntimeError(f"Bedrock embedding failed: {e}")
    
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")
    print("FAISS index saved to disk.")
    return vector_store

if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)


