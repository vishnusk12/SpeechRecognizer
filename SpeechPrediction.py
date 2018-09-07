# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:05:05 2018

@author: hp
"""
from preprocess import get_labels, wav2mfcc
import numpy as np
from keras.models import model_from_json

feature_dim_1 = 20
feature_dim_2 = 11
channel = 1

json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]

prediction = predict('./data/bed/004ae714_nohash_1.wav', model)