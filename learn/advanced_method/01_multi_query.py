# -*- coding: utf-8 -*-
# @Time : 2025/8/18 20:29
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Dm411X7H1?t=291.7
# @File : 01_multi_query.py
# @Software: PyCharm 
# @Comment :
'''
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
conda config --set custom_channels.auto https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
conda install databases  -y

进阶RAG检索 MultiQueryRetriever
准备工作（加载数据、定义embedding模型、向量库）
'''
import warnings
warnings.filterwarnings('ignore')
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# load pdf
loader = PyPDFLoader("../../data/baichuan.pdf")
data = loader.load()
# split

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)
splits = text_splitter.split_documents(data[:6])

import os
from getpass import getpass

OPENAI_API_KEY = getpass()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ['OPENAI_BASE_URL'] = 'https://api.openai-hk.com/v1'
# VectorDB
embedding = OpenAIEmbeddings()
vectorDB = Chroma.from_documents(
    documents=splits,
    embedding=embedding
)

# MultiQueryRetriever
# Set logging for the queries
# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

question = "what is baichuan2 ?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorDB.as_retriever(),
    llm=llm
)
docs = retriever_from_llm.get_relevant_documents(query=question)

print(docs)
print('3' * 100)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """基于以下提供的内容回答问题，如果内容中不包含问题的答案，请回答“我不知道”
内容：
{contexts}

问题： {query}
"""

mulitquery_PROMPT = PromptTemplate(
    input_variables=["query", "contexts"], template=template,
)  # Chain
qa_chain = LLMChain(llm=llm, prompt=mulitquery_PROMPT)
out = qa_chain(inputs={"query": question,
                       "contexts": "\n---\n".join([d.page_content for d in docs])}
               )
print(out)
