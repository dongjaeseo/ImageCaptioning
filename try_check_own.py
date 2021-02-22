from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM, add, Conv1D
max_length = 34
vocab_size = 1949
embedding_dim = 200

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape = max_length)
se1 = Embedding(vocab_size, embedding_dim)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256, activation = 'relu')(se2)

decoder1 = add([fe2, se3])
# decoder1 = concatenate([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()
