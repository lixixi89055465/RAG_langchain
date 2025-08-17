# -*- coding: utf-8 -*-
# @Time : 2025/8/16 23:18
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Hk4y1X7aG?t=661.0
# @File : embedd.py.py
# @Software: PyCharm 
# @Comment :
from sentence_transformers import SentenceTransformer

# model=SentenceTransformer('moka/m3e-base')
model = SentenceTransformer('/home/nanji/workspace/m3e-base')

# Our sentences we like to encode
sentences = ['为什么良好的睡眠对健康至关重要?',
             '良好的睡眠有助于身体修复自身,增强免疫系统',
             '在监督学习中，算法经常需要大量的标记数据来进行有效学习',
             '睡眠不足可能导致长期健康问题,如心脏病和糖尿病',
             '这种学习方法依赖于数据质量和数量',
             '它帮助维持正常的新陈代谢和体重控制',
             '睡眠对儿童和青少年的大脑发育和成长尤为重要',
             '良好的睡眠有助于提高日间的工作效率和注意力',
             '监督学习的成功取决于特征选择和算法的选择',
             '量子计算机的发展仍处于早期阶段，面临技术和物理挑战',
             '量子计算机与传统计算机不同，后者使用二进制位进行计算',
             '机器学习使我睡不着觉',
             ]
# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
print(len(embeddings))
print('0' * 100)
print(len(embeddings[0]))
from sklearn.manifold import TSNE
import numpy as np

tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

plt.rcParams['axes.unicode_minus'] = False
color_list = ['black'] * len(embeddings_2d[1:])
color_list.insert(0, 'red')

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color=color_list)

for i in range(len(embeddings_2d)):
    plt.text(embeddings_2d[:, 0][i], embeddings_2d[:, 1][i] + 2, sentences[i], color=color_list[i])

# 显示图表
plt.savefig("embedding.png")

from transformers import GPT2TokenizerFast

# tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-embedding-ada-002')
tokenizer = GPT2TokenizerFast.from_pretrained('/home/nanji/workspace/text-embedding-ada-002')
print(tokenizer.encode('hello world'))
assert tokenizer.encode('hello world') == [15339, 1917]
tokenizer.generate("hello")

# model = SentenceTransformer("/home/nanji/workspace/text-embedding-ada-002")
# embeddings = tokenizer.encode(sentences)
embeddings_openai = [tokenizer.encode(item) for item in sentences]
print('2' * 100)
print(embeddings_openai)
embeddings_openai =np.array(embeddings_openai)
tsne = TSNE(n_components=2, perplexity=5)
embeddings_openai_2d = tsne.fit_transform(embeddings_openai)
print(len(embeddings_openai))
print(len(embeddings_openai[0]))
