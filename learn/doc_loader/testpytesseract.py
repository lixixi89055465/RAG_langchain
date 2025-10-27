# -*- coding: utf-8 -*-
# @Time : 2025/10/24 22:50
# @Author : nanji
# @Site : https://blog.csdn.net/u010698107/article/details/121736386
# @File : testpytesseract.py
# @Software: PyCharm
# @Comment :
import pytesseract
from pytesseract import Output

try:
    from PIL import Image
except ImportError:
    import Image

# 列出支持的语言
# print(pytesseract.get_languages(config=''))
# print(pytesseract.image_to_string(Image.open('test.png'), lang='chi_sim+eng'))

img = Image.open('testimg2.png')
print(pytesseract.image_to_boxes(img, output_type=Output.STRING, lang='chi_sim'))
print("#" * 30)
print(pytesseract.image_to_data(img, output_type=Output.STRING, lang='chi_sim'))