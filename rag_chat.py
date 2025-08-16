# -*- coding: utf-8 -*-
# @Time : 2025/8/14 22:57
# @Author : nanji
# @Site : 
# @File : rag_chat.py.py
# @Software: PyCharm 
# @Comment :
from pydantic import BaseModel
# import os
# from langchain_openai import ChatOpenAI
#
# base_url = "http://192.168.11.178:11434/v1"
# chat = ChatOpenAI(
#     # openai_api_key=os.environ['OPENAI_API_KEY'],
#     base_url=base_url,
#     openai_api_key='',
#     # model='gpt-3.5-turbo'
#     model='qwen2:latest'
# )
# print(chat)

import os
from langchain.chat_models import ChatOpenAI

# from langchain_openai import ChatOpenAI

base_url = "http://192.168.11.178:11434/v1"

chat = ChatOpenAI(
    # openai_api_key=os.environ['OPENAI_API_KEY'],
    base_url=base_url,
    openai_api_key='empty',
    # model='gpt-3.5-turbo'
    # model='qwen2:latest',
    model='chevalblanc/gpt-4o-mini:latest'
)
print(chat)

# from langchain.schema import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage
# )
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Knock knock."),
    AIMessage(content="Who's there?"),
    HumanMessage(content="Orange"),
]
res = chat(messages)
print('1' * 100)
print(res)
messages.append(res)
# res = chat(messages)
print('2' * 100)

print(res)
messages = [
    SystemMessage(content="你是一个专业的知识助手。"),
    HumanMessage(content="你知道baichuan2模型吗？"),
]
# res = chat(messages)
# print('3' * 100)
# print(res.content)

baichuan2_information = [
    "Baichuan 2是一个大规模多语言语言模型，它专注于训练在多种语言中表现优异的模型，包括不仅限于英文。这使得Baichuan 2在处理各种语言的任务时能够取得显著的性能提升。",
    "Baichuan 2是从头开始训练的，使用了包括了2.6万亿个标记的庞大训练数据集。相对于以往的模型，Baichuan 2提供了更丰富的数据资源，从而能够更好地支持多语言的开发和应用。",
    "Baichuan 2不仅在通用任务上表现出色，还在特定领域（如医学和法律）的任务中展现了卓越的性能。这为特定领域的应用提供了强有力的支持。"
]

source_knowledge = '\n'.join(baichuan2_information)
print('4' * 100)
print(source_knowledge)

print('5' * 100)
query = '你知道baichuan2模型吗？'
prompt_template = f"""基于以下内容回答问题：

内容:
{source_knowledge}

Query: {query}"""

prompt = HumanMessage(
    content=prompt_template
)
# messages.append(prompt)
# res = chat(messages)
# print('6' * 100)
# print(res.content)
"""
当我们注入一些专业的知识后，模型就能够很好的回答相关问题。 如果每一个问题都去用相关的外部知识进行增强拼接的话，那么回答的准确性就大大增加？？？？
"""
### 创建一个RAG对话模型
#### 1. 加载数据 （以baichuan2论文为例）

# https://arxiv.org/pdf/2309.10305v2.pdf

# ! pip install pypdf

# from langchain.document_loaders import PyPDFLoader,PyPDFium2Loader
from langchain_community.document_loaders import Docx2txtLoader, PyPDFium2Loader, \
    PyPDFLoader, UnstructuredMarkdownLoader, CSVLoader

# loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")
loader = PyPDFium2Loader('data/2309.10305v4.pdf')
pages = loader.load_and_split()
# print('7' * 100)
print(pages[0])

#### 2. 知识切片 将文档分割成均匀的块。每个块是一段原始文本
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
# print('8' * 100)
docs = text_spliter.split_documents(pages)
# print(len(docs))
### 3. 利用embedding模型对每个文本片段进行向量化，并储存到向量数据库中
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# embed_model = OpenAIEmbeddings()
embed_model = OllamaEmbeddings(
    model='nomic-embed-text',
    base_url="http://192.168.11.178:11434",
    # normalize=True
)

# vectorstore = Chroma.from_documents(documents=docs, \
#                                     embedding=embed_model, \
#                                     collection_name="openai_embed")

# 4. 通过向量相似度检索和问题最相关的K个文档
query = "How large is the baichuan2 vocabulary?"


# result = vectorstore.similarity_search(query, k=2)
# print('9' * 100)
# print(result)


#### 5. 原始`query`与检索得到的文本组合起来输入到语言模型，得到最终的回答

def augment_prompt(query: str):
    # 获取top3 的文本片段
    results = vectorstore.similarity_search(query, k=20)
    source_knowledge = '\n'.join([x.page_content for x in results])
    # 构建 prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    contexts:
        {source_knowledge}
  
      query: {query}"""
    return augmented_prompt


# print(augment_prompt(query))
# 创建prompt
# prompt = HumanMessage(
#     content=augment_prompt(query)
# )
# messages.append(prompt)
# res = chat(messages)
print('1' * 100)
# print(res.content)
### 没有OPENAI api key怎么办 创建一个非openai的对话模型

# 1.   embedding模型
# 2.   chat模型

from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.vectorstores import Chroma

from sentence_transformers import SentenceTransformer

# model_name = "sentence-transformers/sentence-t5-large"

# embedding = HuggingFaceEmbeddings(model_name=model_name)
embedding = SentenceTransformer(
    # 'sentence-transformers/sentence-t5-large',
    # 'sentence-t5-large',
    # cache_folder=r'/home/nanji/workspace/sentence-t5-large'
    '/home/nanji/workspace/sentence-t5-large'
)
# vectorstore_hf = Chroma.from_documents(
#     documents=docs,
#     embedding=embedding,
#     collection_name="huggingface_embed"
# )
# from transformers import LlamaForCausalLM
# embedding = LlamaForCausalLM.from_pretrained(
#     "/home/nanji/workspace/sentence-t5-large",
#     local_files_only=True)

vectorstore_hf = Chroma.from_documents(documents=docs, embedding=embedding.encode , collection_name="huggingface_embed")

result = vectorstore_hf.similarity_search(query, k=2)
print('2' * 100)
print(result)
# 通过本地部署的模型进行交互
