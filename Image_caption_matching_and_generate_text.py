from os import listdir
from pickle import dump
import pickle
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
import numpy as np

dir_text = "C:\\Users\\user\\Desktop\\Autism image-text\\Text\\Autism.token.txt"

def extract_features(directory):
    in_layer = Input(shape=(224, 224, 3))
    model = VGG16(include_top=False, input_tensor=in_layer)
    print(model.summary())
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)
        
    return features

model_T = VGG16(include_top=True)
print(model_T.summary())

directory = "C:\\Users\\user\\Desktop\\Autism image-text\\Autism"
features = extract_features(directory)
print('Extracted Features: %d' % len(features))

# save to file
dump(features, open("C:\\\\Users\\\\user\\\\Desktop\\\\Autism image-text\\\\features.pkl", 'wb'))

import string

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) <2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        imege_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping

def to_vacabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

filename = "C:\\\\Users\\\\user\\\\Desktop\\\\Autism image-text\\\\Text\\\\Dis.txt"
doc = load_doc(dir_text)
descriptions=load_descriptions(doc)
print('Loaded: %d' % len(descriptions))

voca = to_vacabulary(descriptions)
print(voca)

save_descriptions(descriptions, filename)

#loading train data

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


# load photo features

def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# create sequences of images, input sequences and output words for an image

#X1 부분이 1x4로 나오니까 그거 다시 봐야 되
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                print(photos[key][0])
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def load_set(filename):
    doc = load_doc(filename)
    dataset=list()
    for line in doc.split('\n'):
        if len(line)<1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.rstrip('.jpg')
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def max_length(dataset_desc):
    lines = to_lines(dataset_desc)
    return max(len(d.split()) for d in lines)

filename_a = "C:\\Users\\user\\Desktop\\Autism image-text\\Text\\Autism.trainID.txt"
train = load_set(filename_a)
print('Dataset: %d' % len(train))

train_descriptions = load_clean_descriptions(filename, train)
print('Descriptions: train=%d' % len(train_descriptions))
print(train_descriptions)

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index)+1
print("Voca size: %d" %  vocab_size)

max_lengh = max_length(train_descriptions)
print("D_len: %d" % max_lengh)

def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

train_features = load_photo_features("C:\\Users\\user\\Desktop\\Autism image-text\\features.pkl",train_descriptions)
print('Photos: train=%d' % len(train_features))

X1train, X2train, ytrain = create_sequences(tokenizer, max_lengh, train_descriptions, train_features, vocab_size)

filename_b = "C:\\Users\\user\\Desktop\\Autism image-text\\Text\\Autism.testID.txt"
test = load_set(filename_b)
print('Dataset: %d' % len(test))

test_descriptions = load_clean_descriptions(filename, test)
print('Descriptions: test=%d' % len(test_descriptions))

test_features = load_photo_features("C:\\Users\\user\\Desktop\\Autism image-text\\features.pkl", test_descriptions)
print('Photos: test=%d' % len(test_features))

X1test, X2test, ytest = create_sequences(tokenizer, max_lengh, test_descriptions, test_features, vocab_size)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(7,7,512))
    flt = Flatten()(inputs1)
    fe1 = Dropout(0.5)(flt)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_lengh,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1=add([fe2, se3])
    decoder2=Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam')
    
    print(model.summary())
    return model

model = define_model(vocab_size, max_length)
filepath = "C:\\Users\\user\\Desktop\\Autism image-text\\Matching.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

plot_model(model, to_file='model.png', show_shapes=True)

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
