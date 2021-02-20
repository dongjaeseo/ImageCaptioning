import numpy as np
import matplotlib.pyplot as plt
import glob
import string
import os
from PIL import Image
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
            words = [word for word in words if len(word)>1]
            image_desc = 'startseq ' + ' '.join(words) + ' endseq'
            
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

train_image_feature = {}
for i, id in enumerate(train_id):
    train_image_feature[id] = image_feature(train_path[i])

val_image_feature = {}
for i, id in enumerate(val_id):
    val_image_feature[id] = image_feature(val_path[i])

# print(val_image_feature['3251906388_c09d44340e'])

# 모델
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()

# 임베딩 레이어!! 우린 이거의 웨이트 값이 변하길 원치 않아욧
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

def data_generator(descriptions, image_feat, word_int, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    while 1:
        for id, captions in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = image_feat[id]
            for line in captions:
                # encode the sequence
                seq = [word_int[word] for word in line.split(' ') if word in word_int]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # out_seq = to_categorical(out_seq, num_classes=vocab_size)
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n=0

cp = ModelCheckpoint('../input/model/prac_val_10.hdf5', save_best_only=True)
lr = ReduceLROnPlateau(factor=0.5, patience = 3, verbose = 1)
es = EarlyStopping(patience = 6)
epochs = 50
batch_size = 3
steps = np.ceil(len(train_descriptions)/batch_size)
val_steps = np.ceil(len(val_descriptions)/batch_size)

train_generator = data_generator(train_descriptions, train_image_feature, word_int, max_length, batch_size)
val_generator = data_generator(val_descriptions, val_image_feature, word_int, max_length, batch_size)
model.fit(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks =[es,lr,cp], validation_data= val_generator, validation_steps=val_steps)
model.save('../input/model/prac_val_10_noes.hdf5')

model_cp = load_model('../input/model/prac_val_10.hdf5')
model_last = load_model('../input/model/prac_val_10_noes.hdf5')

def greedySearch(photo, model):
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

def beam_search_predictions(image, model, beam_index = 3):
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
for path in pics_path:
    image1 = image_feature(path)
    image1 = image1.reshape((1,2048))
    x = plt.imread(path)
    plt.imshow(x)
    plt.show()
    print("Greedy Search:",greedySearch(image1, model_cp))
    print("Greedy Search:",greedySearch(image1, model_last))
    print("Beam Search, K = 3:",beam_search_predictions(image1, model_cp, beam_index = 3))
    print("Beam Search, K = 3:",beam_search_predictions(image1, model_last, beam_index = 3))
    print("Beam Search, K = 5:",beam_search_predictions(image1, model_cp, beam_index = 5))
    print("Beam Search, K = 5:",beam_search_predictions(image1, model_last, beam_index = 5))
    print("Beam Search, K = 7:",beam_search_predictions(image1, model_cp, beam_index = 7))
    print("Beam Search, K = 7:",beam_search_predictions(image1, model_last, beam_index = 7))
    print("Beam Search, K = 10:",beam_search_predictions(image1, model_cp, beam_index = 10))
    print("Beam Search, K = 10:",beam_search_predictions(image1, model_last, beam_index = 10))