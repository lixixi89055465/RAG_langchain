# -*- coding: utf-8 -*-
# @Time : 2025/8/19 23:10
# @Author : nanji
# @Site : 
# @File : 04_long_contexts.py
# @Software: PyCharm 
# @Comment :
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
import os
from getpass import getpass

OPENAI_API_KEY = getpass()
os.environ['OPENAI_BASE_URL'] = 'https://api.openai-hk.com/v1'

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 10})
base_docs = retriever.get_relevant_documents(
    "What is baichuan2 ？"
)
print('0' * 100)
print(base_docs)
from langchain_community.document_transformers import (
    LongContextReorder,
)

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(base_docs)
print('1' * 100)
print(reordered_docs)
