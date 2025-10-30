# -*- coding: utf-8 -*-
# @Time : 2025/10/29 21:05
# @Author : nanji
# @Site : 
# @File : Agent_COT_ReAct.py
# @Software: PyCharm
# @Comment :

from dotenv import load_dotenv
import os

# 记载.env 文件中的环境变量

load_dotenv()
# OpenAI ChatGpt
query = '9.9和9.11哪个数字更大?'
from openai import OpenAI
from getpass import getpass

# client = OpenAI(api_key=os.getenv('OpenAI_key'))
# for i in range(3):
#     response = client.chat.completions.create(
#         model='gpt-4o-mini',
#         messages=[
#             {'role': 'system', 'content': 'You are a helpful assistant.'},
#             {'role': 'user', 'content': query}
#         ]
#     )
#     message = response.choices[0].message.content
#     print(f'第{+ 1} 次 i **** {message}')

# Anthropic Claude
# import anthropic
# CLIENT_ant=anthropic.Anthropic(api_key=os.getenv('claude_key'))


# Cohere
import cohere

# co = cohere.ClientV2(api_key=os.getenv('cohere_key'))
# for i in range(3):
#     response = co.chat(
#         model='command-r',
#         messages=[{"role": "user", "content": query}]
#     )
#     print(f'第{i + 1} 次 **** {response}')
# import cohere
# co = cohere.ClientV2(os.getenv('cohere_key'))
# response = co.chat(
#     model="command-a-03-2025",
#     messages=[{"role": "user", "content": "hello world!"}]
# )
# print(response)

# os.environ['DASHSCOPE_API_KEY'] = '1561484:sk-2041da74e9e043a9a66f2a4f15f65731'
# os.environ['DASHSCOPE_API_KEY'] = '2738218:sk-7693b269a1a44df48973b2dbed8d45be'
# 同意前文
from http import HTTPStatus
from dashscope import Generation

# messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
#             {'role': 'user', 'content': query}]
# for i in range(3):
#     response = Generation.call(
#         model='qwen-turbo',
#         messages=messages,
#         result_format='message',
#         api_key=os.getenv('DASHSCOPE_API_KEY')
#     )
#     # print(f'第{i + 1} 次**** {response.output.choices[0].message.content}')
#     print(f'第{i + 1} 次**** {response}')

import os
from dashscope import Generation
import dashscope

# 若使用新加坡地域的模型，请释放下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '你是谁？'}
    ]
response = Generation.call(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",   # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=messages,
    result_format="message"
)

if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
# # llama 3.1
# from groq import Groq
# llama=Groq(api_key=os.getenv('llama_api_key'))
