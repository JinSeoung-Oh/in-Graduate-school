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

#########################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
X = vectorizer.fit_transform(sentence)

print(X)
print(X.shape)
type(X)
##########################################################################

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
print('shape:', x.shape)

import random

random.shuffle(x)

EMotionLabels = pd.get_dummies(word.values[:, 1])

print(EMotionLabels)

NeedsLabels = pd.get_dummies(word.values[:, 2])

print(NeedsLabels)


vocab_size = len(tokenizer.word_index)+1

print(vocab_size)

from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer
#########################################################################
class Attention_1(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec_1')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state_1')(hidden_states)
        score = dot([score_first_part, h_t], [2,1], name='attention_score_1')
        attention_weights = Activation('softmax', name = 'attention_weight_1')(score)
        context_vector = dot([hidden_states, attention_weights], [1,1,], name='context_vector_1')
        pre_activation = concatenate([context_vector, h_t], name = 'attention_output_1')
        attention_vector = Dense(128, use_bias = False, activation='tanh', name='attention_vector_1')(pre_activation)
        
        return attention_vector

from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer

class Attention_2(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec_2')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state_2')(hidden_states)
        score = dot([score_first_part, h_t], [2,1], name='attention_score_2')
        attention_weights = Activation('softmax', name = 'attention_weight_2')(score)
        context_vector = dot([hidden_states, attention_weights], [1,1,], name='context_vector_2')
        pre_activation = concatenate([context_vector, h_t], name = 'attention_output_2')
        attention_vector = Dense(128, use_bias = False, activation='tanh', name='attention_vector_2')(pre_activation)
        
        return attention_vector
##########################################################################

from keras.layers import Layer
import keras.backend as K


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

from keras.layers import Layer
import keras.backend as K


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

def bulid_EMoBranch(inputs, numEMo, finalAct="softmax"):
    z = Embedding(vocab_size,100, input_length = 20 )(inputs)
    z = Bidirectional(LSTM(100, return_sequences=True))(z)
    z = Bidirectional(LSTM(100, return_sequences=True))(z)
    z = Bidirectional(LSTM(100, return_sequences=True))(z)
    z = Bidirectional(LSTM(100, return_sequences=False))(z)
    z = Dense(256)(z)
    z = Activation("relu")(z)
    z = Dense(numEMo)(z)
    z = Activation(finalAct, name = "Emotion_classification_output")(z)
    
    return z

def bulid_EMoBranch(inputs, numEMo, finalAct="softmax"):
    z = Embedding(vocab_size,100, input_length = 20 )(inputs)
    z = Bidirectional(LSTM(100, return_sequences=True))(z)
    z = Bidirectional(LSTM(100, return_sequences=True))(z)
    z = Bidirectional(LSTM(100, return_sequences=True))(z)
    z = attention_1()(z)
    z = Dense(numEMo)(z)
    z = Activation(finalAct, name = "Emotion_classification_output")(z)
    
    return z

from tensorflow.keras.layers import Input

def bulid_model(numEMo, numNee, finalAct="softmax"):
    
    inputs = tensorflow.keras.Input(shape=(20,))
    EMoBranch = bulid_EMoBranch(inputs, numEMo, finalAct="softmax")
    NeeBranch = bulid_NeeBranch(inputs, numNee, finalAct="softmax")
    
    model = Model(inputs = inputs, outputs =[EMoBranch, NeeBranch], name ="Emotion-Needs")
    
    return model

model = bulid_model(numEMo=5, numNee=8, finalAct = "softmax")

losses ={
    "Emotion_classification_output" : "categorical_crossentropy",
    "Needs_classification_output" : "categorical_crossentropy",
}

lossWeights = {"Emotion_classification_output":0.5, "Needs_classification_output":0.5}


model.compile(optimizer="Adam", loss=losses, loss_weights=lossWeights, metrics = ["accuracy"])
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 20)  # 조기종료 콜백함수

import gc

print("Fitting the model...") 
hist = model.fit(x=x, y={"Emotion_classification_output" : EMotionLabels , "Needs_classification_output" : NeedsLabels}, 
                 batch_size = 20, epochs = 200,callbacks = [early_stopping] ,validation_split = .2) 
print("Model successfully fitted!!")

gc.collect()
