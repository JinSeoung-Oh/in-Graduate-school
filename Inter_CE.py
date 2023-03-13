import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
word = pd.read_excel("C:\\Users\\user\\Desktop\\IR corpus\\IR corpus.xlsx")
sentence = word.values[:, 0]
Emotion_label=word.values[:, 1]
Needs_label=word.values[:, 2]

from sklearn.feature_extraction.text import CountVectorizer
from keras.utils.np_utils import to_categorical
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence)
tokenized_sequences = tokenizer.texts_to_sequences(word.values[:, 0])
x=pad_sequences(tokenized_sequences, padding='post',maxlen=20)

import random

random.shuffle(x)

EMotionLabels = pd.get_dummies(word.values[:, 1])
NeedsLabels = pd.get_dummies(word.values[:, 2])

import tensorflow.keras as keras

base_model=keras.models.load_model("C:\\Users\\user\\Desktop\\IR corpus\\Needs Branch.h5")

base_model._layers.pop()
base_model._layers.pop()
base_model._layers.pop()
base_model._layers.pop()
base_model._layers.pop()
base_model._layers.pop()

base_model.summary()

for i in range(7):
    base_model.layers[i].trainable=False

base_model.summary()

numEmo = 5
z=base_model.get_layer('gru_4').output
z=GRU(100, return_sequences=True, return_state=True, name='gru_5')(z)
z=GRU(100, return_sequences=False, name='gru_6')(z)
z=Dense(256)(z)
z=Activation("relu")(z)
z=Dense(numEmo, activation="softmax", name="Emotion_classification")(z)

from tensorflow.keras.layers import Input

inputs=keras.Input(shape=(20,))
model = Model(inputs=base_model.input, outputs=z, name="predict_Emotion_using_Needs")
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 20)  # 조기종료 콜백함수
import gc

print("Fitting the model...")
model.fit(x=x, y=EMotionLabels, batch_size = 20, epochs = 200,callbacks = [early_stopping]
                ,validation_split = .1)
print("Model successfully fitted!!")     

gc.collect()
