import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import glob

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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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
            # image_desc = ' '.join(tokens[1:])
            words = tokens[1:]
            words = [word.lower() for word in words]
            words = [word.translate(table) for word in words]
            image_desc = ' '.join(words)
            
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
# '1034276567_49bb87c51c': ['a boy bites hard into a treat while he sits outside '

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
    if id in train_id:
        if id not in train_descriptions:
            train_descriptions[id] = []
        for i in range(len(caption)):
            tokens = caption[i].split()
            new_caption = 'startseq ' + ' '.join(tokens) + ' endseq'
            train_descriptions[id].append(new_caption)

val_descriptions = {}
for id, caption in descriptions.items():
    if id in val_id:
        if id not in val_descriptions:
            val_descriptions[id] = []
        for i in range(len(caption)):
            tokens = caption[i].split()
            new_caption = 'startseq ' + ' '.join(tokens) + ' endseq'
            val_descriptions[id].append(new_caption)
# print(train_descriptions) '1032460886_4a598ed535': ['startseq a man is standing in front of a skyscraper endseq',
# print(val_descriptions) '1032460886_4a598ed535': ['startseq a man is standing in front of a skyscraper endseq'

# 반복되는 단어 찾기
all_captions = []
for _, captions in train_descriptions.items():
    for line in captions:
        all_captions.append(line)
        
for _, captions in val_descriptions.items():
    for line in captions:
        all_captions.append(line)
# print(len(all_captions), all_captions[0:10]) 40460 ['startseq a child in a pink dress is climbing up a set of stairs in an entry way endseq'

word_repeat = 10
word_counts = {}
for line in all_captions:
    for w in line.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1 

vocab = [word for word in word_counts if word_counts[word] >= word_repeat]
# print(len(vocab)) 1957

# wordtoix 는 {단어:숫자}
# ixtoword 는 {숫자:단어}
# 1부터 숫자 부여해줌
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

# 전체문장 중 제일 단어 수가 많은 문장의 단어 수 찾기!
num = []
for _, captions in train_descriptions.items():
    for line in captions:
        num.append(len(line.split()))
for _, captions in val_descriptions.items():
    for line in captions:
        num.append(len(line.split()))

max_length = np.max(num)
# print(max_length) 38
embedding_dim = 200

embedding_matrix = pickle.load(open('../input/emb_mat.pkl', 'rb'))
# print(embedding_matrix[word_int['startseq']]) [0 0 0 ]

# 이미지 특성을 찾아주기만 하게 인셉션 v3 에서 마지막 레이어를 빼준다~~!
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

# 이미지를 벡터화 시켜서 1차원으로 만들어줭 이미지 경로 이용
def encode(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fea_vec = model_new.predict(x) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

model = load_model('../input/model/real.hdf5')

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_int[w] for w in in_text.split() if w in word_int]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = int_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_predictions(image, beam_index = 3):
    start = [word_int["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [int_word[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

pics_path = glob.glob('../input/test/' + '*.jpg')
# print(pics_path)
for path in pics_path:
    print(path)
    image1 = encode(path)
    image1 = image1.reshape((1,2048))
    x = plt.imread(path)
    plt.imshow(x)
    plt.show()
    print("Greedy Search:",greedySearch(image1))
    print("Beam Search, K = 3:",beam_search_predictions(image1, beam_index = 3))
    print("Beam Search, K = 5:",beam_search_predictions(image1, beam_index = 5))
    print("Beam Search, K = 7:",beam_search_predictions(image1, beam_index = 7))
    print("Beam Search, K = 10:",beam_search_predictions(image1, beam_index = 10))
# pic = '2398605966_1d0c9e6a20.jpg'
# image = encoding_test[pic].reshape((1,2048))
# x=plt.imread(images_path+pic)
# plt.imshow(x)
# plt.show()

# print("Greedy Search:",greedySearch(image))
# print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
# print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
# print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))
# print("Beam Search, K = 10:",beam_search_predictions(image, beam_index = 10))

# pic = list(encoding_test.keys())[1]
# image = encoding_test[pic].reshape((1,2048))
# x=plt.imread(images_path+pic)
# plt.imshow(x)
# plt.show()

# print("Greedy:",greedySearch(image))
# print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
# print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
# print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))