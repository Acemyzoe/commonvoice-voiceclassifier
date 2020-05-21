#!/usr/bin/python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import warnings

import librosa
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

def mfccprep(mfccs):
    mfcc = {}	
    for idx, val in enumerate(mfccs):
        mfcc['mfcc'+str(idx+1)]=val
    return mfcc

def init_data():
    rows = []
    feature = {}
    #Load the csv into a dataframe and shows the relationship between audio clip and the features like gender and ethinicty of the speaker
    csv = pd.read_csv('./zh-CN/train.tsv',sep='\t')
    print(csv.index)
    #for every file in folder-
    for x in csv.index:
        file_name = './zh-CN/clips/'+str(csv.path[x])
        print(x,file_name)
        #load the mp3 file in this path and retrieves X is audio time series and its sample rate
        X, sample_rate = librosa.load(file_name)
        #retieves mfccs and finds the mean across the 13 mfccs separately
        mfccs = list(pd.DataFrame(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)).T.mean())
        feature = mfccprep(mfccs)
        try:
            feature['age'] = csv.age[x]
        except:
            feature['age']= None
        try:
            feature['gender'] = csv.gender[x]
        except:
            feature['gender']=None
        rows.append(feature)

    data = np.array(rows)
    np.save('data.npy',data)


def load_data():
    data = np.load('data.npy',allow_pickle=True)
    rows = data.tolist()

    #storing all data retrieved into a dataframe
    df = pd.DataFrame.from_dict(rows)
    df = df.dropna()
    df['gender'] = df.gender.apply(lambda x: 1 if x=='male' else 0)
    agekeys = {'thirties':3, 'twenties':2, 'sixties':6, 'fourties':4, 'fifties':5, 'teens':1,
        'seventies':7, 'eighties':8}
    df.age = df.age.apply(lambda x: agekeys[x])
 
    X = df.drop(['gender','age'], axis=1) #
    y = df.gender
    #y = df.age
    
    lb = LabelEncoder()
    #converts labels into categorical data
    y = np_utils.to_categorical(lb.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    num_labels = y.shape[1]
    print('num_labels:',num_labels)

    return X_train, X_test, y_train, y_test

def model():
    # build neural network model
    model = Sequential()

    model.add(Dense(256, input_shape=(13,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    #model.summary()
    return model

def train_model():
    mymodel = model()
    mymodel.summary()
    x_train, x_test, y_train, y_test = load_data()
    #fits the model and validates output with test data.
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = mymodel.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    test_scores = mymodel.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])    
    mymodel.save('gender_model.h5')

def deploy_gender(file_name):
    mymodel = model()
    mymodel.load_weights('./gender_model.h5')
    #mymodel = keras.models.load_model('./gender_model.h5')
    rows = []
    feature = {}
    #load the mp3 file in this path and retrieves X is audio time series and its sample rate
    X, sample_rate = librosa.load(file_name)
    #retieves mfccs and finds the mean across the 13 mfccs separately
    mfccs = list(pd.DataFrame(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)).T.mean())
    feature = mfccprep(mfccs)
    rows.append(feature)
    df = pd.DataFrame.from_dict(rows)
    print(mymodel.predict(df)[0])
    print(mymodel.predict_classes(df)[0])
    
    return mymodel.predict_classes(df)[0]
    
if __name__ == "__main__":
    '''
    train_model()
    '''
    deploy_gender('./zh-CN/clips/common_voice_zh-CN_18531536.mp3') # male    teens
    deploy_gender('./zh-CN/clips/common_voice_zh-CN_19792544.mp3') # male    twenties
    deploy_gender('./zh-CN/clips/common_voice_zh-CN_19703883.mp3') # female    thirties
    
