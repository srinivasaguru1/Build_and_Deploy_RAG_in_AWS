import json
import os
import sys
import boto3
import streamlit as st

from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from QASystem.ingestion import data_ingestion, get_vector_store
from QASystem.retrievalandgeneration import get_llama3_llm, get_response_llm  #  Also fix: you used llama3 but called llama2 later

# Initialize Bedrock and embeddings
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

def main():
    st.set_page_config(page_title="QA with Doc")  # FIX: Proper keyword
    st.header(" QA with Documents using LangChain + AWS Bedrock")

    user_question = st.text_input(" Ask a question from the PDF files")

    with st.sidebar:
        st.title(" Vector Store Management")
        
        if st.button("Update Vectors"):
            with st.spinner(" Processing documents..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success(" Vector store updated")

        if st.button(" Ask with LLaMA3"):
            with st.spinner(" Searching..."):
                # Load FAISS index
                faiss_index = FAISS.load_local(
                    "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
                )

                llm = get_llama3_llm()  #  Corrected function name
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success(" Answer generated")

if __name__ == "__main__":
    main()
