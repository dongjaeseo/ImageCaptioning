import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import string
import os
from PIL import Image
from time import time
from tqdm import tqdm

from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout, add, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

token_path = "../input/flickr8k/Data/Flickr8k_text/Flickr8k.token.txt"
images_path = '../input/flickr8k/Data/Flicker8k_Dataset/'
glove_path = '../input/glove6b'

doc = open(token_path,'r').read()

# 그리고 딕셔너리 형태로 이미지이름:[캡션1,캡션2] 형태로 저장
descriptions = dict()
table = str.maketrans('', '', string.punctuation)
for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split('.')[0]
            words = tokens[1:]
            words = [word.lower() for word in words]
            words = [word.translate(table) for word in words]
            image_desc = 'startseq ' + ' '.join(words) + ' endseq'
            
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
    
print(descriptions)

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

# 트테스
train_descriptions = {}
for id, caption in descriptions.items():
    if id in train_id:
        if id not in train_descriptions:
            train_descriptions[id] = []
        for i in range(len(caption)):
            train_descriptions[id].append(caption)

val_descriptions = {}
for id, caption in descriptions.items():
    if id in val_id:
        if id not in val_descriptions:
            val_descriptions[id] = []
        for i in range(len(caption)):
            val_descriptions[id].append(caption)

# 반복되는 단어 찾기
all_captions = []
for _, captions in descriptions.items():
    for line in captions:
        all_captions.append(line)

word_repeat = 10
word_counts = {}
for line in all_captions:
    for w in line.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1 

vocab = [word for word in word_counts if word_counts[word] >= word_repeat]

int_word = {}
word_int = {}
number = 1
for word in vocab:
    int_word[number] = word
    word_int[word] = number
    number += 1
# print(int_word) 1010: 'museum'
# zero padding 때문
vocab_size = len(int_word) + 1

# max_length
num = []
for _, captions in descriptions.items():
    for line in captions:
        num.append(len(line.split()))
max_length = np.max(num)

embeddings_dict = {} 
glove = open(os.path.join(glove_path,'glove.6B.200d.txt'), encoding="utf-8")
for line in tqdm(glove):
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings_dict[word] = vector

# 글로브에서 벡터 가지고옴 
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, number in word_int.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[number] = embedding_vector

# pickle.dump(embedding_matrix, open('../input/emb_mat.pkl', 'wb'))

model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

# 이미지를 벡터화 시켜서 1차원으로 만들어줭 이미지 경로 이용
def image_feature(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
    x = preprocess_input(x)
    fea_vec = model_new.predict(x)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

train_encoding = {}
for i, id in enumerate(train_id):
    train_encoding[id] = image_feature(train_path[i])
train_features = train_encoding

val_encoding = {}
for i, id in enumerate(val_id):
    val_encoding[id] = image_feature(val_path[i])
val_features = val_encoding

print(val_encoding)