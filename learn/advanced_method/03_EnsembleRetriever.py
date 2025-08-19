# -*- coding: utf-8 -*-
# @Time : 2025/8/19 22:47
# @Author : nanji
# @Site : 
# @File : 03_EnsembleRetriever.py
# @Software: PyCharm 
# @Comment :
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from getpass import getpass

OPENAI_API_KEY = getpass()
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] = 'https://api.openai-hk.com/v1'

embedding = OpenAIEmbeddings()
# Load pdf
loader = PyPDFLoader('../../data/baichuan.pdf')
data = loader.load()
# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data[:6])
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(
    documents=splits
)
bm25_retriever.k = 4

vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
)

docs = ensemble_retriever.invoke("What is baichuan2 ？")
print(docs)
