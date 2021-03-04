ImageCaption Generator Project


Goal is to create a sentence that describes a given image file!

이 프로젝트의 목적은 주어진 사진을 설명해주는 문장(캡션)을 만들어 주는 것입니다!



Basic concept:

The model concept is quite similar to Machine Translation model.

해당 모델의 아이디어는 기계번역(Machine Translation)과 많이 유사합니다.



The basic idea of Machine Translation is
1. Encoder model(RNN) that converts given text into a context vector
2. Decoder model(also an RNN model) that converts context vector to full sentence in another language!
먼저 기계번역의 아이디어에 대해 말씀드리면
1. 인코더(RNN모델)이 인풋 데이터(문장) 을 벡터화 해줍니다. (이렇게 하면 해당 벡터에는 문장의 내용이 들어가게 됩니다.)
2. 디코더 모델(RNN모델) 이 그 내용백터를 다시 다른 언어의 문장으로 만들어줍니다!


And here comes the basic idea of Image Captioning:
If we can make a context vector from image(unique context vector that describes an image!),
we can build a Decoder model to generate sentence from an image! 
여기서 이미지 캡셔닝의 대한 아이디어를 말씀드리면
이미지로부터 그 이미지를 설명해주는 내용벡터를 만들면
디코더 모델을 만들어 그 내용벡터를 문장화 할 수 있습니다!


All we need is a dataset that contains images with captions describing each images.
Flickr8k is one such dataset provided from Kaggle. 
As you can guess from the name, it has approximately 8000 images with 5 captions for each image.(making 40k captions)
단지 필요한건 이미지와 문장들이 담긴 데이터셋입니다!
저는 캐글에서 제공하는 Flickr8k 라는 데이터셋을 사용할겁니다.
데이터셋의 구조는 8천장의 사진과 각 사진을 설명하는 5줄의 문장, 총 40000줄의 문장으로 이루어져있습니다


The first approach to the model is to preprocess the input data(almost about text file)
For the sentences in the text file, the steps are as follows:
1. Removing the punctuations
2. Lower all capital letters in the sentences(This is to prevent the machine from taking 'Apple' and 'apple' as different texts!)
모델에 대한 첫 번째 접근은 전처리입니다!(대부분 텍스트에 대한 전처리입니다)
문장들에 이러한 전처리를 해줍니다
1. 특수문자를 지워주고,
2. 대문자를 소문자로 변환해줍니다!


Some of the words in the text dataset are not repeated enough to make training meaningful
So I set word threshold value of 10 (i.e: judging that words below 10 repetition has no merit in training)
and made a vocab dictionary that contains words with 10+ repetition
Here's the vocab size of my dictionary : 1948

Before going deeper into data, I'll have to explain about zero_padding and Embedding.
Not all the sentences we use have same lengths. So I'm going to PAD the sentences with meaningless 0


