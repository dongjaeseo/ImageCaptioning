from tensorflow.keras.utils import to_categorical

a = to_categorical(3, num_classes=5)
print(a)