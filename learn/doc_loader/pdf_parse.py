# -*- coding: utf-8 -*-
# @Time : 2025/10/24 22:08
# @Author : nanji
# @Site : 
# @File : pdf_parse.py
# @Software: PyCharm
# @Comment :
# pip install pillow Tesseract pdf2image llama_parse pip install nest-asyncio
#  sudo apt  install tesseract-ocr
# sudo apt install libtool
'''
wget http://www.leptonica.org/source/leptonica-1.86.0.tar.gz
sudo apt install libtool pkg-config -y
'''
# 文本解析
# 文本解析器已经很成熟了。它们可以读取文档，并从文件中提取文本。常见的例子包括 PyPDF、PyMUPDF 和 PDFMiner以及很多其他。
from langchain_community.document_loaders import PyPDFLoader

file_path = '../../data/Nvidia_2025.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
print(docs[1].page_content)
# OCR 文本解析
# 如果选择像Pytesseract这样的OCR工具，不仅能更有效地捕获文本，还能保留文档的结构。
# 这种方法比基础的文本解析器能更好地保留原始格式和上下文。

from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

# pytesseract.pytesseract.tesseract_cmd = '/home/nanji/anaconda3/envs/py310/bin/pytesseract'

# 将文件路径作为参数传递
# pages = convert_from_path(file_path)
# all_text = ""

# 确保 i 在页面范围内，避免索引越界
# for i in range(len(pages)):
#     filename = f"page{i}.jpg"
#     pages[i].save(filename, 'JPEG')
#     # 输出文本的文件
#     outfile = f"page{i}_text.txt"
#     # 使用 with 语句打开文件，确保安全关闭
#     with open(outfile, "a") as f:
#         text = str(pytesseract.image_to_string(Image.open(filename), lang="chi_sim"))
#         # 写入文本
#         f.write(text)
#         all_text += text + "\n"  # 每页的文本用换行符分隔
# else:
#     print(f"PDF 只有 {len(pages)} 页，无法访问第 {i + 1} 页")
# 智能文档解析（IDP）
# 一种集成多种技术的文档处理方法，旨在高效地将非结构化文档转换为结构化数据。
# 它可以帮助自动化提取文本和相关信息，
# 并通过诸如OCR、LLM和Markdown格式化等技术来增强解析效果。
import getpass

os.environ["LLAMA_CLOUD_API_KEY"] = getpass.getpass()
from llama_parse import LlamaParse
import nest_asyncio

nest_asyncio.apply()
documents = LlamaParse(result_type="markdown").load_data(file_path)
# print(documents[0].get_content())
# 构建RAG系统
# pip install langchain==0.2.13
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain import PromptTemplate

os.environ['OPENAI_BASE_URL'] = 'https://api.openai-hk.com/v1'
os.environ["OPENAI_API_KEY"] = getpass.getpass()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

Question: {question} 

Context: {context} 

Answer:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question']
)
# # 文本解析 RAG 表现
docs = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    docs,
    OpenAIEmbeddings(),
    collection_name='pyparse_db'
)
base_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
# rag_chain = (
#         {'context': base_retriever, 'question': RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
# )
# r1 = rag_chain.invoke("2024财年第一季度的营业收入是多少？")
# r1 = rag_chain.invoke("2024财年第四季度的营业收入是多少？")
# print(r1)
# 根据提供的上下文，2024财年第一季度的营业收入没有被提到。因此，我不知道2024财年第一季度的营业收入是多少。

## OCR RAG 表现
# text_ocr = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(docs,
#                                     OpenAIEmbeddings(),
#                                     collection_name="pyparse_db")
# base_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
#
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# rag_chain=(
#     {'context':base_retriever,'question':RunnablePassthrough()}
#     |prompt
#     |llm
#     |StrOutputParser()
# )
# rag_chain.invoke("2024财年第一季度的营业收入是多少？")
# IDP
docs_idp = text_splitter.split_text(documents[0].get_content())
vectorstore_idp = Chroma.from_texts(
    docs_idp,
    OpenAIEmbeddings(),
    collection_name='pyparse_idp',
)
base_retriever_idp = vectorstore_idp.as_retriever(search_kwargs={'k': 3})
rag_chain = (
        {'context': base_retriever_idp, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
r2 = rag_chain.invoke("2024财年第一季度的营业收入是多少？")
print(r2)
