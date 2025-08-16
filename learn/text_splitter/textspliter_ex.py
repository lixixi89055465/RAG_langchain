# -*- coding: utf-8 -*-
# @Time : 2025/8/16 17:17
# @Author : nanji
# @Site : 
# @File : textspliter_ex.py
# @Software: PyCharm 
# @Comment :

'''
文本分块：当加载完一个文件之后，需要对文件进行分割，切割逻辑可以按照：
章节、段落、句子、词组等
上下文长度的限制，LLM通常有固定的输入长度限制（4K、8K 、 16K、32K等）
GPT4 turbo 128K
提高处理效率，减少响应时间，
提高输出质量，LLM处理较短的文本块时，通常能够生成更精确、更相关的输出。这是因为模型更容易理解和响应较小的文本范围内的复杂性和细节。
TextSplitter的工作 流程
将文本拆分成小的、语义上有意义的块（通常是句子）。
将这些小块组合成一个较大的块，直到达到一定的大小chunksize（通过某种函数来衡量,一般是长度）。
一旦达到这个大小，就将这个大块视为一个独立的文本片段，然后开始创建一个新的文本块，并保留一些重叠部分（chunkoverlap以保持块与块之间的上下文联系）。

'''
from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
孔乙己
孔乙己是站着喝酒而穿长衫的唯一的人。他身材很高大;青白脸色，绉纹间时常夹些伤痕;
一部乱蓬蓬的花白的胡子。穿的虽然是长衫，可是又脏又破，似乎十多年没有补，也没有洗。
 他对人说话，总是满口之乎者也，教人半懂不懂的。因为他姓孔，
 别人便从描红纸上的“上大人孔乙己”这半懂不懂的话里，替他取下一个绰号，叫作孔乙己。
孔乙己一到店，所有喝酒的人便都看着他笑，有的叫道，“孔乙己，你脸上又添上新伤疤了!”
他不回答，对柜里说，“温两碗酒，要一碟茴香豆。”便排出九文大钱。他们又故意的高声嚷道，
“你一定又偷了人家的东西了!”孔乙己睁大眼睛说，“你怎么这样凭空污人清白……”“什么清白?
我前天亲眼见你偷了何家的书，吊着打。” 孔乙己便涨红了脸，额上的青筋条条绽出，争辩道，
“窃书不能算偷……窃书!……读书人的事，能算偷么?”接连便是难懂的话，什么“君子固穷”，
什么“者乎”之类，引得众人都哄笑起来：店内外充满了快活的空气。
听人家背地里谈论，孔乙己原来也读过书，但终于没有进学，又不会营生;
于是愈过愈穷，弄到将要讨饭了。 幸而写得一笔好字，便替人家钞钞书，换一碗饭吃。
可惜他又有一样坏脾气，便是好喝懒做。坐不到几天，便连人和书籍纸张笔砚，一齐失踪。
如是几次，叫他钞书的人也没有了。孔乙己没有法，便免不了偶然做些偷窃的事。
但他在我们店里，品行却比别人都好，就是从不拖欠;虽然间或没有现钱，暂时记在粉板上，
但不出一月，定然还清，从粉板上拭去了孔乙己的名字。
'''
text = '''孔乙己

孔乙己是站着喝酒而穿长衫的唯一的人。他身材很高大;青白脸色，绉纹间时常夹些伤痕;一部乱蓬蓬的花白的胡子。穿的虽然是长衫，可是又脏又破，似乎十多年没有补，也没有洗。
他对人说话，总是满口之乎者也，教人半懂不懂的。因为他姓孔，别人便从描红纸上的“上大人孔乙己”这半懂不懂的话里，替他取下一个绰号，叫作孔乙己。

孔乙己一到店，所有喝酒的人便都看着他笑，有的叫道，“孔乙己，你脸上又添上新伤疤了!”他不回答，对柜里说，“温两碗酒，要一碟茴香豆。”便排出九文大钱。他们又故意的高声嚷道，“你一定又偷了人家的东西了!”孔乙己睁大眼睛说，“你怎么这样凭空污人清白……”“什么清白?我前天亲眼见你偷了何家的书，吊着打。”
孔乙己便涨红了脸，额上的青筋条条绽出，争辩道，“窃书不能算偷……窃书!……读书人的事，能算偷么?”接连便是难懂的话，什么“君子固穷”，什么“者乎”之类，引得众人都哄笑起来：店内外充满了快活的空气。听人家背地里谈论，孔乙己原来也读过书，但终于没有进学，又不会营生;于是愈过愈穷，弄到将要讨饭了。
幸而写得一笔好字，便替人家钞钞书，换一碗饭吃。可惜他又有一样坏脾气，便是好喝懒做。坐不到几天，便连人和书籍纸张笔砚，一齐失踪。如是几次，叫他钞书的人也没有了。孔乙己没有法，便免不了偶然做些偷窃的事。但他在我们店里，品行却比别人都好，就是从不拖欠;虽然间或没有现钱，暂时记在粉板上，但不出一月，定然还清，从粉板上拭去了孔乙己的名字。'''
print(text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
)
chunks = text_splitter.split_text(text)

print(len(chunks))

for i, _ in enumerate(chunks):
    print(f'chunk #{i} , size :{len(chunks[i])}')
    print(chunks[i])
    print('-' * 100)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    length_function=len
)
chunks = text_splitter.split_text(text)
print('1' * 100)
print(len(chunks))
for i, _ in enumerate(chunks):
    print(f'chunk #{i}, size:{len(chunks[i])}')
    print(chunks[i])

    print('-' * 100)

# chunksize ==100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
)
chunks = text_splitter.split_text(text)
print('2' * 100)
for i, _ in enumerate(chunks):
    print(f'chunk #{i}, size:{len(chunks[i])}')
    print(chunks[i])
    print('-' * 100)
print(len(chunks))

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    separators=['\n\n', '\n', '。', ''],
    keep_separator=True,
    is_separator_regex=True
)
chunks = text_splitter.split_text(text)
print(len(chunks))
print('2' * 100)
for i, _ in enumerate(chunks):
    print(f'chunk #{i}, size:{len(chunks[i])}')
    print(chunks[i])
    print('-' * 100)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    separators=['\n\n', '\n', '。', ''],
    keep_separator=False,
    is_separator_regex=True
)
chunks = text_splitter.split_text(text)
print(len(chunks))
print('3' * 100)
for i, _ in enumerate(chunks):
    print(f'chunk #{i}, size:{len(chunks[i])}')
    print(chunks[i])
    print('-' * 100)
