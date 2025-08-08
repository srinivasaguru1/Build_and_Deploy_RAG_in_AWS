from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
import boto3

# Initialize AWS Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Embedding model from Bedrock
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock_client
)

# Prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words 
with detailed explanations. If you don't know the answer, 
just say that you don't know — don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Function to get the LLM from Bedrock
def get_llama3_llm():
    llm = BedrockLLM(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock_client
    )
    return llm

# Function to generate response using RetrievalQA
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({"query": query})  # ✅ Replaced deprecated `__call__` with `invoke`
    return answer["result"]

# Main execution
if __name__ == '__main__':
    # Load FAISS index
    faiss_index = FAISS.load_local(
        "faiss_index",
        bedrock_embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Query
    query = "What is a rag?"

    # Get LLM
    llm = get_llama3_llm()

    # Generate and print answer
    print(get_response_llm(llm, faiss_index, query))
