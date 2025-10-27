# -*- coding: utf-8 -*-
# @Time : 2025/10/25 10:01
# @Author : nanji
# @Site :  https://blog.csdn.net/u013172930/article/details/147860521
# @File : test_nest_asyncio.py
# @Software: PyCharm
# @Comment :
# pip install pillow pytesseract pdf2image llama_parse pip install nest-asyncio
import nest_asyncio

print(nest_asyncio)
import asyncio
import nest_asyncio

# 修补 asyncio
nest_asyncio.apply()

async def inner_task():
    await asyncio.sleep(1)
    print("Inner task completed")

async def outer_task():
    print("Outer task started")
    await inner_task()
    print("Outer task completed")

# 在已有事件循环中运行
loop = asyncio.get_event_loop()
loop.run_until_complete(outer_task())
