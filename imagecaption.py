# model = InceptionV3(weights='imagenet')
# model_new = Model(model.input, model.layers[-2].output)

# # 이미지를 벡터화 시켜서 1차원으로 만들어줭 이미지 경로 이용
# def encode(image_path):
#     img = image.load_img(image_path, target_size=(299, 299))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     fea_vec = model_new.predict(x) 
#     fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
#     return fea_vec

# class model:
#     def __init__(self, model):
#         self.modeltype = model
        
#     def encode(self, image_path):
#         model = self.modeltype(weights = 'imagenet')
#         model_new = Model(model.input, model.layers[-2].output)