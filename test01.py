# -*- coding: utf-8 -*-
# @Time : 2025/8/19 22:00
# @Author : nanji
# @Site : 
# @File : test01.py
# @Software: PyCharm 
# @Comment :
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("/home/nanji/workspace/baichuan-inc/Baichuan2-7B-Chat",
                                          use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/nanji/workspace/baichuan-inc/Baichuan2-7B-Chat",
                                             device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/home/nanji/workspace/baichuan-inc/Baichuan2-7B-Chat")
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)