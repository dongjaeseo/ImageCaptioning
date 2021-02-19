from tensorflow.keras.utils import to_categorical
import numpy as np
# a = to_categorical(3, num_classes=5)
# print(a)

# start = [0]
# start_word = [start,0]
# print(start_word)

from sklearn.model_selection import train_test_split
# data = set([1,2,3,4,5])
# data = np.array(data)
# x,y = train_test_split(data,train_size = 0.8)
# print(x)

import numpy as np
import matplotlib.pyplot as plt

import string
import os
from PIL import Image
from time import time

from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout, add, concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

token_path = "../input/flickr8k/Data/Flickr8k_text/Flickr8k.token.txt"
train_images_path = '../input/flickr8k/Data/Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_path = '../input/flickr8k/Data/Flickr8k_text/Flickr_8k.testImages.txt'
images_path = '../input/flickr8k/Data/Flicker8k_Dataset/'
glove_path = '../input/glove6b'

doc = open(token_path,'r').read()
# print(doc[:410])

# 캡션+이미지 사진을 줄별로 잘라준다
# len(line) > 2 는 마지막에 빈 줄때문에 있다
# token = line.split() 하면 줄이 다 나눠진다
# image_id 에 tokens[0] 를 .jpg 빼주고
# 띄워놓은 image_desc 를 다시 붙여줌 스페이스로
# 그리고 딕셔너리 형태로 이미지이름:[캡션1,캡션2] 형태로 저장
# '1034276567_49bb87c51c': ['a boy bites hard into a treat while he sits outside '
descriptions = dict()
table = str.maketrans('', '', string.punctuation)
for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split('.')[0]
            # image_desc = ' '.join(tokens[1:])
            words = tokens[1:]
            words = [word.lower() for word in words]
            words = [word.translate(table) for word in words]
            image_desc = ' '.join(words)
            
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)

# 트레인 테스트 스플릿
image_id = []
for keys,_ in descriptions.items():
    if keys not in image_id:
        image_id.append(keys)

train_id, val_id = train_test_split(image_id,train_size = 0.8, random_state = 42)
# print(train_id[0]) 297285273_688e44c014

# 경로 끌어오기
train_path = []
for id in train_id:
    train_path.append(f'../input/flickr8k/Data/Flicker8k_Dataset/{id}.jpg')

val_path = []
for id in val_id:
    val_path.append(f'../input/flickr8k/Data/Flicker8k_Dataset/{id}.jpg')

# 시퀀스 시작과 끝 찍어준다
train_descriptions = {}
for id, caption in descriptions.items():
    if id not in train_descriptions:
        train_descriptions[id] = []
    for i in range(len(caption)):
        tokens = caption[i].split()
        new_caption = 'startseq ' + ' '.join(tokens) + ' endseq'
        train_descriptions[id].append(new_caption)

val_descriptions = {}
for id, caption in descriptions.items():
    if id not in val_descriptions:
        val_descriptions[id] = []
    for i in range(len(caption)):
        tokens = caption[i].split()
        new_caption = 'startseq ' + ' '.join(tokens) + ' endseq'
        val_descriptions[id].append(new_caption)

# print(train_descriptions) '1032460886_4a598ed535': ['startseq a man is standing in front of a skyscraper endseq',
# print(val_descriptions) '1032460886_4a598ed535': ['startseq a man is standing in front of a skyscraper endseq'

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

# 그래도 10번은 반복되어야 의미있는 단어 아니겠소?
# 10번도 반복안되면 쓰지 말자!!! 한번 숫자 바꿔보자!!
# 딕셔너리 워드마다 숫자 늘어남!!
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]