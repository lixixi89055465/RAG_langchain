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

# 同意前文
from http import HTTPStatus
from dashscope import Generation

messages = [{'role': 'system', 'content': 'You are a helful assistant.'},
            {'role': 'user', 'content': query}]
for i in range(3):
    response = Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message',
        api_key=os.getenv('DASHSCOPE_API_KEY')
    )
    # print(f'第{i + 1} 次**** {response.output.choices[0].message.content}')
    print(f'第{i + 1} 次**** {response}')
