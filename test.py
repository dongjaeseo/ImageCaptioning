import numpy as np
import matplotlib.pyplot as plt
import glob
import string

from tensorflow.keras import Input, layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout, add, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

token_path = "../input/flickr8k/Data/Flickr8k_text/Flickr8k.token.txt"
images_path = '../input/flickr8k/Data/Flicker8k_Dataset/'
glove_path = '../input/glove6b'

doc = open(token_path,'r').read()

descriptions = dict()
table = str.maketrans('', '', string.punctuation)
for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split('.')[0]
            words = tokens[1:]
            words = [word.lower() for word in words]
            words = [word.translate(table) for word in words]
            words = [word for word in words if len(word)>1]
            image_desc = 'start ' + ' '.join(words) + ' end'
            
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

# 트테스
train_descriptions = {}
for id, caption in descriptions.items():
    if id in train_id:
        if id not in train_descriptions:
            train_descriptions[id] = []
        for line in caption:
            train_descriptions[id].append(line)

val_descriptions = {}
for id, caption in descriptions.items():
    if id in val_id:
        if id not in val_descriptions:
            val_descriptions[id] = []
        for line in caption:
            val_descriptions[id].append(line)

# 10 번 이상 반복되는 단어 수 세기!
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
# print(len(vocab))  >> 1948

# 토큰화
int_word = {}
word_int = {}
number = 1
for word in vocab:
    int_word[number] = word
    word_int[word] = number
    number += 1

# zero padding 을 포함하기 위해 1을 더해준다
vocab_size = len(int_word) + 1

# max_length
num = []
for _, captions in descriptions.items():
    for line in captions:
        num.append(len(line.split()))
max_length = np.max(num)

# GloVe 를 불러온다!
embeddings_dict = {} 
glove = open(('../input/glove6b/glove.6B.200d.txt'), encoding="utf-8")
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings_dict[word] = vector

# embedding_matrix 에 현재 vocab 에 대한 GloVe 벡터를 불러온다!
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, number in word_int.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[number] = embedding_vector

# InceptionV3 사용, 이미지의 특성 찾기
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

# 이미지의 특징이 담긴 벡터를 반환하는 함수
def image_feature(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
    x = preprocess_input(x)
    fea_vec = model_new.predict(x)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

train_image_feature = {}
for id in train_id:
    train_image_feature[id] = image_feature(f'../input/flickr8k/Data/Flicker8k_Dataset/{id}.jpg')

val_image_feature = {}
for id in val_id:
    val_image_feature[id] = image_feature(f'../input/flickr8k/Data/Flicker8k_Dataset/{id}.jpg')

# 모델
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256, activation = 'relu')(se2)

decoder1 = add([fe2, se3])
# decoder1 = concatenate([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()

# 임베딩 레이어에 가중치를 저장! 
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model = load_model('../input/model/imagecaption.hdf5')

def greedySearch(photo, model):
    in_text = 'start'
    for i in range(max_length):
        sequence = [word_int[w] for w in in_text.split() if w in word_int]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = int_word[yhat]
        in_text += ' ' + word
        if word == 'end':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_predictions(image, model, beam_index = 3):
    start = [word_int["start"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [int_word[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'end':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

pics_path = glob.glob('../input/test2/' + '*.jpg')
for path in pics_path:
    image1 = image_feature(path)
    image1 = image1.reshape((1,2048))
    x = plt.imread(path)
    plt.imshow(x)
    plt.show()
    print("Greedy:",greedySearch(image1, model))
    print("Beam, 3:",beam_search_predictions(image1, model, beam_index = 3))
    print("Beam, 5:",beam_search_predictions(image1, model, beam_index = 5))
    print("Beam, 7:",beam_search_predictions(image1, model, beam_index = 7))
    print("Beam, 10:",beam_search_predictions(image1, model, beam_index = 10))