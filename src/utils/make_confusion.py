import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.models import model_from_json
import pickle 
import os
import librosa
import matplotlib.pyplot as plt
import sys

from dataloaders import SpectogramDataLoader

LENGTH=3 * 44100

warnings.resetwarnings()

def get_test_data(test_dir, labels_dict):
    X_test = {}
    y_test = {}
    mappings = ["Bar", "Cla", "Rom", "Mod"]

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".npy"):
                x_ = np.load(os.path.join(root, file))
                x_id = os.path.splitext(file)[0]
                X_test[x_id] = x_
                y_test[x_id] = mappings.index(labels_dict[int(x_id)])
    return X_test, y_test

def test(model, X_test, y_test):
    num_corr = 0
    for id, source in X_test.items():
        splitted = [X_test[id][i * LENGTH : (i + 1) * LENGTH] for i in range((len(X_test[id]) + LENGTH) // LENGTH)][:-1]
        label = y_test[id]
        processed = []
        for sentence in range(len(splitted)):
            processed.append(np.log((np.abs(librosa.stft(splitted[sentence], hop_length=512, win_length=2048))**2) + sys.float_info.epsilon))
        prediction = model.predict(np.array(processed))
        counts = np.sum(prediction, 0)
        predict_label = np.argmax(counts)
        if predict_label == label:
            num_corr += 1
            # print("right! prediction: " + str(predict_label) + ", label: " + str(label))
        # else:
        #     print("wrong! prediction: " + str(predict_label) + ", label: " + str(label))
    return float(num_corr) / len(X_test)

def confusion_test(model, X_test, y_test):
    num_corr = 0
    conf = np.zeros(4, 4)
    for id, source in X_test.items():
        splitted = [X_test[id][i * LENGTH : (i + 1) * LENGTH] for i in range((len(X_test[id]) + LENGTH) // LENGTH)][:-1]
        label = y_test[id]
        processed = []
        for sentence in range(len(splitted)):
            processed.append(np.log((np.abs(librosa.stft(splitted[sentence], hop_length=512, win_length=2048))**2) + sys.float_info.epsilon))
        prediction = model.predict(np.array(processed))
        counts = np.sum(prediction, 0)
        predict_label = np.argmax(counts)
        conf[label, predict_label] += 1
        if predict_label == label:
            num_corr += 1
            # print("right! prediction: " + str(predict_label) + ", label: " + str(label))
        # else:
        #     print("wrong! prediction: " + str(predict_label) + ", label: " + str(label))
    return float(num_corr) / len(X_test), conf

# load the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

pwd = '/run/media/estbergm/GSINGH98'
data_path = pwd + os.sep + 'data/'

X_test, y_test = get_test_data(test_dir=data_path + os.sep + 'test/', labels_dict=labels)
acc, confusion = confusion_test(model, X_test, y_test)

print(acc)
print(confusion)
