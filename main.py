import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Embedding,Flatten, Reshape
from tensorflow.keras.layers import LSTM, Bidirectional, GRU
from tensorflow.keras.models import load_model, save_model
from matplotlib import pyplot
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.keras.layers import Embedding,Flatten,Conv1D,MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sys
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import datetime
import h5py

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class attention_1(Layer):
    def __init__(self,**kwargs):
        super(attention_1,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention_1, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention_1,self).get_config()

class attention_2(Layer):
    def __init__(self,**kwargs):
        super(attention_2,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention_2, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention_2,self).get_config()

import pandas as pd
word = pd.read_excel("C:\\Users\\user\\Desktop\\IR corpus\\IR corpus_609.xlsx")
sentence = word.values[:, 0]

le = preprocessing.LabelEncoder()
word['Emotion Label'] = le.fit_transform(word['Emotion Label'])
EMotionLabels = tensorflow.keras.utils.to_categorical(word['Emotion Label'])
EMotionLabels.shape

le = preprocessing.LabelEncoder()
word['Needs Label'] = le.fit_transform(word['Needs Label'])
NeedsLabels = tensorflow.keras.utils.to_categorical(word['Needs Label'])
NeedsLabels.shape

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence)
word_index=tokenizer.word_index

#불용어 정의 및 제거
stopwords = ['이', '를', '의', '을', '그', '꼭','뻔', '이제']

temp_x=[]
tokenized_data=[]
for sentence in word_index:
    temp_x=[word for word in word_index if not word in stopwords]
    tokenized_data.append(temp_x)
    
tokenized_sequences = tokenizer.texts_to_sequences(word.values[:, 0])
x=pad_sequences(tokenized_sequences, maxlen=20)

import random

random.shuffle(x)

vocab_size = len(tokenizer.word_index)+1
from tensorflow.keras.layers import Input

def bulid_EMoBranch(inputs, numEMo, finalAct="softmax"):
    z = Embedding(vocab_size,100, input_length = 20 , name='Ee')(inputs)
    z = Bidirectional(LSTM(100, return_sequences=True, name = 'EB_1'))(z)
    z = attention_1()(z)
    z = Dense(numEMo)(z)
    z = Activation(finalAct, name = "attention_vector_for_Emo")(z)
    
    return z

def bulid_NeeBranch(inputs, numNee, finalAct="softmax"):
    y = Embedding(vocab_size,100, input_length = 20, name='Ne' )(inputs)
    y = Bidirectional(LSTM(100, return_sequences=True, name = 'NB_1'))(y)
    y = attention_2()(y)
    y = Dense(numNee)(y)
    y = Activation(finalAct, name = "attention_vector_for_Nee")(y)
    
    return y

def bulid_model(numEMo, numNee, finalAct="softmax"):
    
    inputs = tensorflow.keras.Input(shape=(20,))
    EMoBranch = bulid_EMoBranch(inputs, numEMo, finalAct="softmax")
    NeeBranch = bulid_NeeBranch(inputs, numNee, finalAct="softmax")
    
    model = Model(inputs = inputs, outputs =[EMoBranch, NeeBranch], name ="For_extract_attention")
    
    return model

model_FA = bulid_model(numEMo=5, numNee=8, finalAct = "softmax")

losses ={
    "attention_vector_for_Emo" : "categorical_crossentropy",
    "attention_vector_for_Nee" : "categorical_crossentropy",
}

lossWeights = {"attention_vector_for_Emo":0.5, "attention_vector_for_Nee":0.5}


model_FA.compile(optimizer="Adam", loss=losses, loss_weights=lossWeights, metrics = ["accuracy"])
model_FA.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 20)  # 조기종료 콜백함수

import gc

print("Fitting the model...") 
hist = model_FA.fit(x, y={"attention_vector_for_Emo" : EMotionLabels , "attention_vector_for_Nee" : NeedsLabels}, 
                 batch_size = 20, epochs = 200,callbacks = [early_stopping] ,validation_split = .2 ) 
print("Model successfully fitted!!")

gc.collect()
###########################################################

from os import listdir
from pickle import dump
import pickle
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
import numpy as np

from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout, Flatten
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

from keras.models import load_model

tokenizer = load(open("C:\\Users\\user\\Desktop\\Autism image-text\\Text\\Train_token.pkl", 'rb'))
max_lengh=27
model =load_model("C:\\Users\\user\\Desktop\\Autism image-text\\Matching.h5")

test_image = extract_features("C:\\Users\\user\\Desktop\\Autism image-text\\test")

from numpy import argmax

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_lengh):          #photo = dict
    in_text = 'startseq'
    for i in range(max_lengh):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_lengh) #numpy.ndarray
        print(sequence)
        print(photo)
        yhat = model.predict([photo, sequence], verbose=0)       #photo의 차원을 왜 늘리려고 하지?
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

description_te = generate_desc(model, tokenizer, test_image, max_lengh)
print(description_te)
#################################################################
for layer in model_FA.layers:
    print(layer.name)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(description_te)
word_index=tokenizer.word_index

    
tokenized_sequences = tokenizer.texts_to_sequences(description_te)
xy=pad_sequences(tokenized_sequences, maxlen=20)

print("extracting E_feature...")

intermediate_layer_model_E = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_1').output)
EF = intermediate_layer_model_E.predict(xy)
print(EF)
EF.shape

print("extracting N_feature...")
intermediate_layer_model_N = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_2').output)
NF = intermediate_layer_model_N.predict(xy)
print(NF)
NF.shape 

NF_size=len(NF)
EF_size=len(EF)

intermediate_layer_model_Elabel = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_vector_for_Emo').output)

YE = intermediate_layer_model_Elabel.predict(xy)

intermediate_layer_model_Nlabel = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_vector_for_Nee').output)

YN = intermediate_layer_model_Nlabel.predict(xy)

text_model = load_model("C:\\Users\\user\\Desktop\\Autism image-text\\Text classification.h5")
xz = NF,EF

print("extracting feature...")
(NeedsProba, EmotionProba) = text_model.predict(xz)

NeedsIdx = NeedsProba[0].argmax()
EmotionIdx = EmotionProba[0].argmax()

Emotion_class = np.array(['행복', '슬픔','분노','당혹감','두려움'])
Needs_class = np.array(['위로', '놀이 지속', '일과 준수', '계속', '보호자', '귀가', '거부', '지속'])

encoder=LabelEncoder()

encoder.fit(Emotion_class)
EmotionLabel = encoder.transform(Emotion_class)

encoder.fit(Needs_class)
NeedsLabel = encoder.transform(Needs_class)

Emotion = encoder.inverse_transform([EmotionIdx])
Needs = encoder.inverse_transform([NeedsIdx])

Autism_Emotion_state = "Autism_Emotion_state: {} ({:.2f}%)".format(Emotion,EmotionProba[0][EmotionIdx]*100)
print(Autism_Emotion_state)

Autism_Needs_state = "Autism_Needs_state: {} ({:.2f}%)".format(Needs,NeedsProba[0][NeedsIdx]*100)
print(Autism_Needs_state)
