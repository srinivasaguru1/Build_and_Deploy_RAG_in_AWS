from langchain_community.embeddings import BedrockEmbedding

from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain.vectorstores import FAISS
