# -*- coding: utf-8 -*-
# @Time : 2025/8/18 22:56
# @Author : nanji
# @Site : 
# @File : 02_contextual-compression.py
# @Software: PyCharm 
# @Comment : https://www.bilibili.com/video/BV1bA4m1573e?t=1.6
## 进阶RAG检索   Contextual Compression + Filtering
### 前期准备
# %%

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load pdf
loader = PyPDFLoader('../../data/baichuan.pdf')
data = loader.load()
# Split

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data[:6])
from getpass import getpass

OPENAI_API_KEY = getpass()
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] = 'https://api.openai-hk.com/v1'
# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectordb.as_retriever()

base_docs = retriever.get_relevant_documents(
    'What is baichuan2 ?'
)
print(base_docs)
# Contextual Extractor

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
compressed_docs = compression_retriever.get_relevant_documents(
    "What is baichuan2 ？"
)

print(compressed_docs)
