# -*- coding: utf-8 -*-
# @Time : 2025/8/16 14:33
# @Author : nanji
# @Site : 
# @File : pdf_loader.py.py
# @Software: PyCharm 
# @Comment :
'''
Langchain 可以使用文档加载器加载不同的文档类型，
如：csv、txt 、html、json以及pdf等，
今天如何分享一下基于pdf的loader
'''
# https://langchain-fanyi.readthedocs.io/en/latest/modules/indexes/document_loaders.html
# 需要解决的问题
# 解析图片，表格
# 页面结构问题
# 格式结构问题(符合人类)

### 使用`pypdf`解析pdf，pdf将按照`page`逐页解析

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('/home/nanji/workspace/RAG_langchain/data/baichuan.pdf')
page_pypdf = loader.load()
# print('0' * 100)
# print(len(page_pypdf))
# pages_pypdf = loader.load()
# print('1' * 100)
# print(len(pages_pypdf))
#
# print('2' * 100)
# print(pages_pypdf[2].page_content)
# print('3' * 100)
# print(pages_pypdf[2].page_content)
#
# print('4' * 100)
# print(pages_pypdf[0].page_content[1583:2000])
#
# pages_pypdf[0].page_content[1583:2000]
# 提取图片信息
## 提取图片信息
# ! pip install rapidocr-onnxruntime
loader = PyPDFLoader(
    '/home/nanji/workspace/RAG_langchain/data/baichuan.pdf',
    extract_images=True)

pages_pypdf_image = loader.load()
# print('5' * 100)
# print(pages_pypdf_image[2].page_content)

# 使用 pyplumber 将pdf逐页进行解析，
# 但是文本结构在分栏的时候存在混淆，解析不完全
from langchain.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader('/home/nanji/workspace/RAG_langchain/data/baichuan.pdf')
data_plumber = loader.load()

# print('6' * 100)
# print(len(data_plumber))
# print('7' * 100)
# print(data_plumber[2].page_content)
# 使用 `PDFMiner`  ，将整个文档解析成一个完整的文本。
# 文本结构可以自行认为定义
from langchain.document_loaders import PDFMinerLoader

loader = PDFMinerLoader('/home/nanji/workspace/RAG_langchain/data/baichuan.pdf')
data_miner = loader.load()

print('8' * 100)
# print(data_miner[0].page_content[1590:1800])
# 使用非结构化 Unstructured
from langchain.document_loaders import UnstructuredPDFLoader

loader = PDFMinerLoader('/home/nanji/workspace/RAG_langchain/data/baichuan.pdf')

data_unstru = loader.load()
# print(data_unstru)
# print('9' * 100)
# print(data_unstru[0].page_content[1662:2000])
# print('1' * 100)
# print(len(data_unstru))
# 非结构化加载器针对不同的文本块创建了不同的“元素”。默认情况下，
# 我们将它们组合在一起，但您可以通过指定 `mode=elements` 轻松保持这种分离。
# 然后依据自己的逻辑进行分离

loader = UnstructuredPDFLoader('/home/nanji/workspace/RAG_langchain/data/baichuan.pdf',
                               mode='elements')
data_elements = loader.load()
print('2'*100)
print(data_elements)
