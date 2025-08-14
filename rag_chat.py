# -*- coding: utf-8 -*-
# @Time : 2025/8/14 22:57
# @Author : nanji
# @Site : 
# @File : rag_chat.py.py
# @Software: PyCharm 
# @Comment :

# import os
# from langchain_openai import ChatOpenAI
#
# base_url = "http://192.168.11.178:11434/v1"
# chat = ChatOpenAI(
#     # openai_api_key=os.environ['OPENAI_API_KEY'],
#     base_url=base_url,
#     openai_api_key='',
#     # model='gpt-3.5-turbo'
#     model='qwen2:latest'
# )
# print(chat)

import os
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

base_url = "http://192.168.11.178:11434/v1"

chat = ChatOpenAI(
    # openai_api_key=os.environ['OPENAI_API_KEY'],
    base_url=base_url,
    openai_api_key='empty',
    # model='gpt-3.5-turbo'
    # model='qwen2:latest',
    model='chevalblanc/gpt-4o-mini:latest'
)
print(chat)

# from langchain.schema import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage
# )
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Knock knock."),
    AIMessage(content="Who's there?"),
    HumanMessage(content="Orange"),
]
res = chat(messages)
print('1' * 100)
print(res)
messages.append(res)
res = chat(messages)
print('2' * 100)

print(res)
messages=[
    SystemMessage(content="你是一个专业的知识助手。"),
    HumanMessage(content="你知道baichuan2模型吗？"),
]
res=chat(messages)
print('3' * 100)
print(res.content)