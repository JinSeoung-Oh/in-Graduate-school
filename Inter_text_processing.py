import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import tensorflow as tf
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
import sys
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import datetime
import h5py

import pandas as pd
word = pd.read_excel("C:\\Users\\user\\Desktop\\IR corpus\\IR corpus_409.xlsx")

sentence = word.values[:, 0]
Emotion_label=word.values[:, 1]
Needs_label=word.values[:, 2]

#tokenize
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

EMotionLabels = pd.get_dummies(word.values[:, 1])
NeedsLabels = pd.get_dummies(word.values[:, 2])

vocab_size = len(tokenizer.word_index)+1

print(vocab_size)

#Emotion_GRU
base_E_model=tensorflow.keras.models.load_model("C:\\Users\\user\\Desktop\\IR corpus\\Emotion Branch model.h5")
base_E_model._layers.pop()
base_E_model._layers.pop()
base_E_model._layers.pop()
base_E_model._layers.pop()
base_E_model._layers.pop()
base_E_model._layers.pop()

#Needs_GRU
base_N_model= tensorflow.keras.models.load_model("C:\\Users\\user\\Desktop\\IR corpus\\Needs Branch.h5")
base_N_model._layers.pop()
base_N_model._layers.pop()
base_N_model._layers.pop()
base_N_model._layers.pop()
base_N_model._layers.pop()
base_N_model._layers.pop()

for i in range(1,4):
    base_E_model.layers[i].trainable=False
    
for i in range(1,7):
    base_N_model.layers[i].trainable=False

for layer in base_E_model.layers:
    print(layer.name)
    
for layer in base_N_model.layers:
    print(layer.name)

base_E_model.summary()
base_N_model.summary()

numNee=8
numEmo=5

#Needs classification uisn Emotion
y=base_E_model.get_layer('Layerlayer_1').output
y=GRU(100, return_sequences=False, name='ESpecial_layer')(y)         
y=Dense(256)(y)
y=Activation("relu")(y)
y=Dense(numNee)(y)
y= Activation("softmax", name="Needs_classification")(y)

#Emotion classification using Needs
z=base_N_model.get_layer('Baselayer_5').output
z=GRU(100, return_sequences=True, return_state=True, name='NSpecial_layer_1')(z)
z=GRU(100, return_sequences=False, name='NSpecial_layer_2')(z)
z=Dense(256)(z)
z=Activation("relu")(z)
z=Dense(numEmo)(z)
z=Activation("softmax", name="Emotion_classification")(z)

#add_LSTM_for_Needs
y=base_E_model.get_layer('baselayer_1').output
y=LSTM(100, return_sequences=True, name='ESpecial_layer_2')(y)
y=LSTM(100, return_sequences=False, name='ESpecial_layer_3')(y)
y=Dense(256)(y)
y=Activation("relu")(y)
y=Dense(numNee)(y)
y= Activation("softmax", name="Needs_classification")(y)


#add_LSTM_for_Emotion
z=base_N_model.get_layer('Baselayer_3').output
z=LSTM(100, return_sequences=True, return_state=True, name='NSpecial_layer_1')(z)
z=LSTM(100, return_sequences=True, return_state=True, name='NSpecial_layer_2')(z)
z=LSTM(100, return_sequences=True, return_state=True, name='NSpecial_layer_3')(z)
z=LSTM(100, return_sequences=False, name='NSpecial_layer_4')(z)
z=Dense(256)(z)
z=Activation("relu")(z)
z=Dense(numEmo)(z)
z=Activation("softmax", name="Emotion_classification")(z)


from tensorflow.keras.layers import Input
inputs = tensorflow.keras.Input(shape=(20,))
model = Model(inputs = [base_E_model.input,base_N_model.input],
                  outputs =[y,z], name ="Emotion-Needs")
losses ={
    "Emotion_classification" : "categorical_crossentropy",
    "Needs_classification" : "categorical_crossentropy",
}

lossWeights = {"Emotion_classification":0.5, "Needs_classification":0.5}

model.compile(optimizer="Adam", loss=losses, loss_weights=lossWeights, metrics = ["accuracy"])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 20)  # 조기종료 콜백함수

import gc
Ndata = x
Edata =x
y1 = EMotionLabels
y2 = NeedsLabels

print("Fitting the model...") 
hist = model.fit([Ndata, Edata], y={"Needs_classification" : y2, "Emotion_classification" : y1}, 
                 batch_size = 20, epochs = 200,callbacks = [early_stopping] ,
                 validation_split=.3)                      

print("Model successfully fitted!!")

gc.collect()

##########################################
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
