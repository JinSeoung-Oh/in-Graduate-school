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
type(x)

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

vocab_size = len(tokenizer.word_index)+1

print(vocab_size)

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

import tensorflow as tf

def focal_loss(gamma=2. , alpha=.25):
    
    gamma = float(gamma)
    alpha = float(alpha)
    
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis = 1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

model_FA = bulid_model(numEMo=5, numNee=8, finalAct = "softmax")

losses ={
    "attention_vector_for_Emo" : focal_loss(),    
    "attention_vector_for_Nee" : focal_loss()
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

tensorflow.keras.utils.plot_model(model_FA, "E_N.png", show_shapes=True)

for layer in model_FA.layers:
    print(layer.name)

#extract att_E_feature
intermediate_layer_model_E = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_1').output)
EF = intermediate_layer_model_E.predict(x)
print(EF)
EF.shape

#extract att_N_feature
intermediate_layer_model_N = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_2').output)
NF = intermediate_layer_model_N.predict(x)
print(NF)
NF.shape 

NF_size=len(NF)
EF_size=len(EF)

intermediate_layer_model_Elabel = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_vector_for_Emo').output)

YE = intermediate_layer_model_Elabel.predict(x)

intermediate_layer_model_Nlabel = Model(inputs=model_FA.input,
                                  outputs=model_FA.get_layer('attention_vector_for_Nee').output)

YN = intermediate_layer_model_Nlabel.predict(x)


def bulid_EMoBranch_N(inputs, numEMo, finalAct="softmax"):
    z = Embedding(NF_size,100, input_length = 20 , name='NF')(inputs)
    z = Bidirectional(LSTM(100, return_sequences=True, name = 'NF_E_1'))(z)
    z = Bidirectional(LSTM(100, return_sequences=True, name = 'NF_E_2'))(z)
    z = Bidirectional(LSTM(100, return_sequences=False, name = 'NF_E_3'))(z)
    z = Dense(numEMo)(z)
    z = Activation(finalAct, name = "Emotion_classification_output_using_NF")(z)
    
    return z

def bulid_NeeBranch_E(inputs, numNee, finalAct="softmax"):
    y = Embedding(EF_size,100, input_length = 20, name='EF' )(inputs)
    y = Bidirectional(LSTM(100, return_sequences=True, name = 'EF_N_1'))(y)
    y = Bidirectional(LSTM(100, return_sequences=True, name = 'EF_N_2'))(y)
    y = Bidirectional(LSTM(100, return_sequences=True, name = 'EF_N_3'))(y)
    y = Bidirectional(LSTM(100, return_sequences=True, name = 'EF_N_4'))(y)
    y = Bidirectional(LSTM(100, return_sequences=True, name = 'EF_N_5'))(y)
    y = Bidirectional(LSTM(100, return_sequences=False, name = 'EF_N_6'))(y)
    y = Dense(numNee)(y)
    y = Activation(finalAct, name = "Needs_classification_output_using_EF")(y)
    
    return y

def bulid_model(numEMo, numNee, finalAct="softmax"):
    
    inputs = tensorflow.keras.Input(shape=(200,))
    EMoBranch = bulid_EMoBranch_N(inputs, numEMo, finalAct="softmax")
    NeeBranch = bulid_NeeBranch_E(inputs, numNee, finalAct="softmax")
    
    model= Model(inputs = inputs, outputs =[EMoBranch, NeeBranch], name ="Emotion-Needs")
    
    return model

model = bulid_model(numEMo=5, numNee=8, finalAct = "softmax")

losses ={
    "Emotion_classification_output_using_NF" : focal_loss(),
    "Needs_classification_output_using_EF" : focal_loss()
}

lossWeights = {"Emotion_classification_output_using_NF":0.5, "Needs_classification_output_using_EF":0.5}


model.compile(optimizer="Adam", loss=losses, loss_weights=lossWeights, metrics = ["accuracy"])
model.summary()

print("Fitting the model...") 
hist_Q = model.fit(x=[NF, EF], y={"Emotion_classification_output_using_NF" : YE , "Needs_classification_output_using_EF" : YN}, 
                 batch_size = 20, epochs = 200,callbacks = [early_stopping] ,validation_split = .2) 
print("Model successfully fitted!!")


gc.collect()
