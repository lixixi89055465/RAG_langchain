# -*- coding: utf-8 -*-
# @Time : 2025/10/17 16:00
# @Author : nanji
# @Site : 
# @File : chunk_size_stu.py
# @Software: PyCharm 
# @Comment :

# 确定文本切割的最优策略
# 在使用基于检索的生成模型（RAG）处理长文本数据时，
# 合理的文本切割策略是提高模型性能和效率的关键。
# 文本切割策略主要依赖于两个参数：chunksize（块大小）和overlap（重叠）。
# 正确配置这些参数可以显著影响模型的输出质量和处理速度。
# chunk_size 基于模型的限制(embedding , LLM )
# 不同Text splitter 的优劣，如何选取
# 可视化文本切分的效果，供大家切分文本初步参考


## 基于模型选取**chunk_size** <a id="**chunk_size**"></a>

# *首先是 ** embedding model **， 向量嵌入模型有 ** Max Tokens ** 的限制，
# 设置的 ** chunk size ** 不可以超过模型支持的最大长度，否则将丢失语义。

# 不同的**embedding model** 支持的 **Max Tokens**都有不同，
# 具体可参考[model 排行](https://huggingface.co/spaces/mteb/leaderboard)

# * 其次是**LLM model** , 大语言模型有**Max sequence length**的限制，
# 处理知识增强的时候，**prompt**中召回的文本不可以超出最大长度。

# 需要根据不同的LLM支持的最大token长度，选取合适的参数
# **不同的文本切分策略**
# * **1: [CharacterTextSplitter](#CharacterTextSplitter)** - 这是最简单的方法。它默认基于字符（默认为""）来分割，并且通过字符的数量来衡量块的长度。
# * **2：[RecursiveCharacterTextSplitter](#RecursiveCharacterTextSplitter)** - 基于字符列表拆分文本。
# * **3: [Document Specific Splitting](#DocumentSpecific)** - 基于不同的文件类型使用不同的切分 (PDF, Python, Markdown)
# * **4: [Semantic Splitting](#SemanticChunking)** - 基于滑动窗口的语义切分
# 那我们就开始实际看一下不同的textsplitter切分效果如何？
#
text = "大家好，我是果粒奶优有果粒，欢迎关注我，让我们一起探索AI。"
### 1 CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator='',
    chunk_size=5,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False
)
a = text_splitter.split_text(text)
# print(a)
# 切分原理
# 创建chunks维护切分的文本块
chunks = []
chunk_size = 5
chunk_overlap = 1
i = 0
while i < len(text):
    # 如果这不是第一块，就回溯chunk_overlap个字符以创建重叠
    if i > 0:
        start = max(i - chunk_overlap, 0)
    else:
        start = i
    # 确定这个块的结束位置
    end = min(start + chunk_size, len(text))
    # 提取块并添加到列表
    chunk = text[start:end]
    chunks.append(chunk)
    # 更新下一块的开始位置
    i = end
# print(chunks)

### 2 RecursiveCharacterTExtSplitter
# RecursiveCharacterTextSplitter, 文本分割工具的设计目的是为了在处理文本时
# 能够在不损失语义关联性的前提下，将文本有效分割成更小的单元。
# 通过先尝试分割段落，如果段落仍然过大，再尝试分割成句子，依此类推，直至分割成单词。
# 这种分割方法尽量保留文本的原有结构和意义，使得处理后的文本单元在语义上保持连贯性。
# * "\n\n" - 段落
# * "\n" - 换行
# * " " - 空格
# * "" - 字符

text = '''  
为什么文本切割在RAG中很重要？RAG（Retrieval-Augmented Generation）是一种将检索机制集成到生成式语言模型中的技术，目的是通过从大量文档或知识库中检索相关信息来增强模型的生成能力。这种方法特别适用于需要广泛背景知识的任务，如问答、文章撰写或详细解释特定主题。在RAG架构中，文本切割（即将长文本分割成较短片段的过程）非常重要，原因如下：

1. **提高检索效率：** 对于大规模的文档库，直接在整个库上执行检索任务既不切实际也不高效。通过将长文本切割成较短的片段，可以使检索过程更加高效，因为短文本片段更容易被比较和索引。这样可以加快检索速度，提高整体性能。

2. **提升结果相关性：** 当查询特定信息时，与查询最相关的内容往往只占据文档中的一小部分。通过文本切割，可以更精确地匹配查询和文档片段之间的相关性，从而提高检索到的信息的准确性和相关性。这对于生成高质量、相关性强的回答尤为重要。

3. **内存和处理限制：** 当代的语言模型，尽管强大，但处理长文本时仍受到内存和计算资源的限制。将长文本分割成较短的片段可以帮助减轻这些限制，因为模型可以分别处理这些较短的文本片段，而不是一次性处理整个长文档。

4. **提高生成质量：** 在RAG框架中，检索到的文本片段将直接影响生成模块的输出。通过确保检索到高质量和高相关性的文本片段，可以提高最终生成内容的质量和准确性。

5. **适应性和灵活性：** 文本切割允许模型根据需要处理不同长度的文本，增加了模型处理各种数据源的能力。这种灵活性对于处理多样化的查询和多种格式的文档特别重要。

总之，文本切割在RAG中非常重要，因为它直接影响到检索效率、结果的相关性、系统的处理能力，以及最终生成内容的质量和准确性。通过优化文本切割策略，可以显著提升RAG系统的整体性能和用户满意度。'''
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=50,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
    separators=['\n\n', '\n', ' ', '']
)

chunk_doc = text_splitter.create_documents([text])
# print(chunk_doc)

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd

Embedding_name = '/home/nanji/workspace/bge-large-zh-v1.5'
# Embedding_name = 'D:/workspace/bge-large-zh-v1.5'
# a = SentenceTransformer(Embedding_name).max_seq_length
# print(a)

# print('1' * 100)
# print(chunk_doc[1].page_content)
tokenizer = AutoTokenizer.from_pretrained(Embedding_name)


# print('2' * 100)
# a = len(tokenizer.encode(chunk_doc[1].page_content))
# print(a)


# token数量
def plot_chunk(chunk_doc, Embedding_name):
    tokenizer = AutoTokenizer.from_pretrained(Embedding_name)
    length = [
        len(tokenizer.encode(doc.page_content)) for doc in chunk_size
    ]
    fig = pd.Series(length).hist()
    plt.show()


# plot_chunk(chunk_doc, Embedding_name)
### 3 其他结构的文本切割
# * **python**    - RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
# * **json**      - RecursiveJsonSplitter
# * **Markdown**  - MarkdownTextSplitter
# * **Html**      - HTMLHeaderTextSplitter

### 4 Semantic Chunking <a id="SemanticChunking"></a>
# 为什么我们在处理文本时通常会使用固定的分块大小，而不考虑实际内容的语义意义。a
# 是不是可以基于文本的语义实现一种更好的方法来处理文本分块，即并非固定参数(**chunksize**)，
# 而是基于语义自行动态确定参数。
# 我们可以通过**embedding**技术进行动态规划
# **Embedding**将文本转化为高维空间中的向量的技术，这些向量能够反映出文本的语义内容。
# 通过文本嵌入技术，可以捕捉到文本的深层次语义信息。当比较两段文本的嵌入向量时，
# 可以根据它们在高维空间中的距离或者角度，来推断这两段文本在语义上的相似度或者差异。利用相似度，
# 将语义上相似的文本自动分组在一起，形成聚类，这有助于更好地理解和组织大量的文本数据。
with open('dream.txt') as file:
    essay = file.read()
#### 我们需要将文本进行拆分，拆分成多个单句，可以按照标点符号进行切分
split_char = ['.', '?', '!']
import re

# Splitting the essay on '.', '?', and '!'
single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
# print(f'{len(single_sentences_list)} sentences were found')
# print('3'*100)
# print(single_sentences_list)
# 我们需要为单个句子拼接更多的句子，但是 list 添加比较困难。因此将其转换为字典列表（List[dict]）
# { 'sentence' : XXX  , 'index' : 0}

sentences = [
    {'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)
]


# print('' * 100)
# print(sentences[:3])

#

def combine_sentences(sentences, buffer_size=1):
    #
    combined_sentences = [
        ' '.join(
            sentences[j]['sentence'] for j in range(
                max(i - buffer_size, 0),

                min(i + buffer_size + 1, len(sentences))))
        for i in range(len(sentences))
    ]
    # 更新原始字典列表，添加组合后的句子
    for i, combined_sentence in enumerate(combined_sentences):
        sentences[i]['combined_sentence'] = combined_sentence
    return sentences


sentences = combine_sentences(sentences)
# print('5' * 100)
# print(sentences[:6])

# 接下来使用**embedding model**对**sentences** 进行编码
from langchain.embeddings import OpenAIEmbeddings
import os
from getpass import getpass

# 加载环境变量
import os
import os

os.environ['OPENAI_BASE_URL'] = 'https://api.openai-hk.com/v1'
os.environ["OPENAI_API_KEY"] = getpass()
oaiembeds = OpenAIEmbeddings()
embeddings = oaiembeds.embed_documents(
    [x['combined_sentence'] for x in sentences]
)
# 将embedding添加到sentence中
for i, sentence in enumerate(sentences):
    sentence['combined_sentence_embedding'] = embeddings[i]

print(sentences[0]['combined_sentence_embedding'])
# 接下来需要根据余弦相似度进行切分
import numpy as np


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


cosine_similarity(sentences[0]['combined_sentence_embedding'], sentences[1]['combined_sentence_embedding'])


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        # calcuate cosine similarity
        similarity = cosine_similarity(embedding_current, embedding_next)
        # Convert to cosine distance
        distance = 1 - similarity
        distances.append(distance)
        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance
    return distances, sentences


distances, sentences = calculate_cosine_distances(sentences)
a = sentences[-2]['distance_to_next']
print(a)
import matplotlib.pyplot as plt

plt.plot(distances)
# 有很多方法可以基于这些距离来划分论文，但我打算将任何超过距离95百分位数的距离视为一个分割点。这是我们需要配置的唯一参数。
import numpy as np

plt.plot(distances)
y_upper_bound = 0.15
plt.ylim(0, y_upper_bound)
plt.xlim(0, len(distances))
# We need to get the distance threshold that we'll consider an outlier
# We'll use numpy .percentile() for this
breakpoint_percentile_threshold = 95

breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
# Then we'll get the index of the distances that are above the threshold.
# This will tell us where we should split our text
indices_above_thresh = [i for i, x in enumerate(distances) if
                        x > breakpoint_distance_threshold]  # The indices of those breakpoints on your list
# Start of the shading and text
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, breakpoint_index in enumerate(indices_above_thresh):
    start_index = 0 if i == 0 else indices_above_thresh[i - 1]
    end_index = breakpoint_index if i <= len(indices_above_thresh) - 1 else len(distances)
    plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
    plt.text(x=np.average([start_index, end_index]),
             y=breakpoint_distance_threshold + (y_upper_bound) / 20,
             s=f"Chunk #{i}", horizontalalignment='center',
             rotation='vertical')

# # Additional step to shade from the last breakpoint to the end of the dataset
# if indices_above_thresh:
#     last_breakpoint = indices_above_thresh[-1]
#     if last_breakpoint < len(distances):
#         plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)],
#                     alpha=0.25)
#         plt.text(x=np.average([last_breakpoint, len(distances)]),
#                  y=breakpoint_distance_threshold + (y_upper_bound) / 20,
#                  s=f"Chunk #{i + 1}",
#                  rotation='vertical')
# plt.title("Essay Chunks Based On Embedding Breakpoints")
# plt.xlabel("Index of sentences in essay (Sentence Position)")
# plt.ylabel("Cosine distance between sequential sentences")
# plt.show()

# Initialize the start index
start_index = 0

# Create a list to hold the grouped sentences
chunks = []

# Iterate through the breakpoints to slice the sentences

for index in indices_above_thresh:
    # The end index is the current breakpoint
    end_index = index
    # Slice the sentence_dicts from the current start index to the end index
    group = sentences[start_index:end_index + 1]
    combined_text = ' '.join([d['sentence'] for d in group])
    chunks.append(combined_text)
    # Update the start index for the next group
    start_index = index + 1
# The last group, if any sentences remain
if start_index<len(sentences):
    combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
    chunks.append(combined_text)

# grouped_sentences now contains the chunked sentences
for i, chunk in enumerate(chunks[:2]):
    buffer = 200
    print (f"Chunk #{i}")
    print (chunk[:buffer].strip())
    print ("...")
    print (chunk[-buffer:].strip())
    print ("\n")