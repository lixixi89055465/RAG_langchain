'''
https://www.bilibili.com/video/BV1ia4y1y7gF?t=310.1

'''
# Huggingface
# Huggingface提供了两种方式调用LLM

# 通过Api token 的方式
# 本地加载
## 使用API  token 调用LLM
# from getpass import getpass
# HUGGINGFACEHUB_API_TOKEN = getpass()

import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from getpass import getpass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
from modelscope import snapshot_download, Model

# model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='master',
#                               # cache_dir='/home/nanji/workspace/Baichuan2-7B-Chat'
#                               cache_dir='/home/nanji/workspace'
#                               )
# # model = Model.from_pretrained("/home/nanji/workspace/Baichuan2-7B-Chat", device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
# model = Model.from_pretrained(model_dir,
#                               device_map="auto",
#                               # allow_remote=False,
#                               # trust_remote_code=True,
#                               torch_dtype=torch.float16)

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/baichuan-inc/Baichuan2-7B-Chat",
                                          use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/nanji/workspace/baichuan-inc/Baichuan2-7B-Chat",
                                             device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/home/nanji/workspace/baichuan-inc/Baichuan2-7B-Chat")
# messages = []
# messages.append({"role": "user", "content": "解释一下“温故而知新”"})
# response = model.chat(tokenizer, messages)
# print(response)

messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)
print('2' * 100)
content = '''Using the contexts below, answer the query.

contexts:
have taken both these aspects into account. We
have expanded the vocabulary size from 64,000
in Baichuan 1 to 125,696, aiming to strike a
balance between computational efficiency and
model performance.
Tokenizer V ocab Size Compression Rate ↓
LLaMA 2 32,000 1.037
Bloom 250,680 0.501
improve after training on more than 2.6 trillion
tokens. By sharing these intermediary results,
we hope to provide the community with greater
insight into the training dynamics of Baichuan 2.
Understanding these dynamics is key to unraveling
the inner working mechanism of large language
Baichuan 2: Open Large-scale Language Models
Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Chao Yin, Chenxu Lv, Da Pan
Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai
Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji

query: How large is the baichuan2 vocabulary size?
'''
messages = []
messages.append({
    "role": "user",
    "content": content
})
response = model.chat(tokenizer, messages)
print(response)
