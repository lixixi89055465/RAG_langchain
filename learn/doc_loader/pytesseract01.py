# -*- coding: utf-8 -*-
# @Time : 2025/10/25 0:39
# @Author : nanji
# @Site : 
# @File : pytesseract01.py
# @Software: PyCharm
# @Comment :
# pip install opencv-python  # 仅包含主要模块
# pip install opencv-contrib-python  # 包含额外模块
# pip install opencv-contrib-python-headless


import numpy as np
import pytesseract
from pytesseract import Output
import cv2

try:
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
except ImportError:
    import Image

img = cv2.imread('testimg2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

width_list = []
for c in cnts:
    _, _, w, _ = cv2.boundingRect(c)
    width_list.append(w)
wm = np.median(width_list)

tess_text = pytesseract.image_to_data(img, output_type=Output.DICT, lang='chi_sim')
for i in range(len(tess_text['text'])):
    word_len = len(tess_text['text'][i])
    if word_len > 1:
        world_w = int(wm * word_len)
        (x, y, w, h) = (tess_text['left'][i], tess_text['top'][i], tess_text['width'][i], tess_text['height'][i])
        cv2.rectangle(img, (x, y), (x + world_w, y + h), (255, 0, 0), 1)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font="simsun.ttc", size=18, encoding="utf-8")
        draw.text((x, y - 20), tess_text['text'][i], (255, 0, 0), font=font)
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# cv2.imshow("TextBoundingBoxes", img)
# cv2.waitKey(0)
cv2.imwrite("aa.png", img)
