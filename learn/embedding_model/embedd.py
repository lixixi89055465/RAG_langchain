# -*- coding: utf-8 -*-
# @Time : 2025/8/16 23:18
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Hk4y1X7aG?t=661.0
# @File : embedd.py.py
# @Software: PyCharm 
# @Comment :
from sentence_transformers import SentenceTransformer

# model=SentenceTransformer('moka/m3e-base')
model=SentenceTransformer('/home/nanji/workspace/m3e-base')


#Our sentences we like to encode
sentences =['为什么良好的睡眠对健康至关重要?' ,
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
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
print(len(embeddings))
print('0'*100)
print(len(embeddings[0]))
