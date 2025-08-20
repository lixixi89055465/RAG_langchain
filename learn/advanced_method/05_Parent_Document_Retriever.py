# -*- coding: utf-8 -*-
# @Time : 2025/8/20 17:11
# @Author : nanji
# @Site : 
# @File : 05_Parent_Document_Retriever.py
# @Software: PyCharm 
# @Comment :
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore

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
### 检索完整的文档
# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name='full_documents', embedding_function=OpenAIEmbeddings()
)

# The storage layer for the parent document
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter
)

retriever.add_documents(data[:6], ids=None)

a = len(list(store.yield_keys()))
print(a)

sub_docs = vectorstore.similarity_search("What is baichuan2 ？")
print(sub_docs)
b = len(sub_docs[0].page_content)
print(b)
retrieved_docs = retriever.get_relevant_documents("What is baichuan2 ？")

c = len(retrieved_docs[0].page_content)
print(c)
len(retrieved_docs)
# 检索较大的文本块
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name='split_parents',
    embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(data[:6], ids=None)
d = len(list(store.yield_keys()))
sub_docs = vectorstore.similarity_search("What is baichuan2 ？")
e = len(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("What is baichuan2 ？")
f = len(retrieved_docs[0].page_content)
print(f)
