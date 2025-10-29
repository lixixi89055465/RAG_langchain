# -*- coding: utf-8 -*-
# @Time : 2025/10/27 23:33
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1rVsTeyEZG?t=180.8
# @File : rag_glm.py
# @Software: PyCharm
# @Comment : pip reinstall unstructured
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_community.document_loaders import UnstructuredPDFLoader
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"

file_path = '../../data/GPU_Programming_Guide_Chinese.pdf'
loader = UnstructuredPDFLoader(file_path)
data = loader.load()

# split
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
all_splits = text_splitter.split_documents(data)

from langchain_community.embeddings import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')
# embeddings = HuggingFaceEmbeddings(model_name='/home/nanji/workspace/m3e-base')
# 本地加载
embeddings = HuggingFaceEmbeddings(
    cache_folder='/home/nanji/workspace/m3e-base',
    model_name='moka-ai/m3e-base'
)


# 创建想粮库
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
# 构建RAG prompt
prompt_template = """你是一个专业的知识问答助手，基于以下内容回答问题，如果无法根据提供的内容回答问题，就直接说'不知道。' "
内容:
{source_knowledge}
Query: {query}"""


def augment_prompt(query: str):
    # 获取top4 的文本片段
    results = vectorstore.similarity_search(query, k=4)
    source_knowledge = '\n'.join([x.page_content for x in results])
    return source_knowledge


prompt = prompt_template.format(
    source_knowledge=augment_prompt(query='什么是动态分支功能?'),
    query='什么是动态分支功能?'
)
print(prompt)
