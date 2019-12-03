from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
import pickle 
import os
import librosa

SR=44100
sentence_length = 60
BATCH_SIZE=15
WORD_SIZE = 4 * SR
EPOCH = 3

# given a raw numpy array, split to 1 minute sentence 
def wav2sentences(nums, label):
  length = sentence_length * SR
  final = [nums[i * length:(i + 1) * length] for i in range((len(nums) + length - 1) // length )][:-1]
  if len(final) >= 1:
    labels = [label for i in range(len(final))]
    final = np.array(final)
    final = final.reshape(final.shape[0], -1, WORD_SIZE)
    return final, labels
  return [], []

# target_path is the csv/pickle file that strores the trackID -> label

def process_data(src, target_path):
  X = []
  y = []
  with open(target_path, 'rb') as handle:
    id2label = pickle.load(handle)
  for root, dirs, files in os.walk(src):
    for file in files:
      x, sr = librosa.load(os.path.join(root, file), sr=SR)
      filename, file_ext = os.path.splitext(file)
      X_temp, y_temp = wav2sentences(x, id2label[int(filename)])
      for x in X_temp:
        X.append(x)
      for label in y_temp:
        y.append(label)
  d = {"data": X, "label": y}
  with open("data.pickle", 'wb') as handle:
    pickle.dump(d, handle)


# process_data("./musicnet/wav", "./musicnet/musicnet_labels.pkl")

with open("./data.pickle", 'rb') as handle:
  d = pickle.load(handle)

X = d["data"]
y = d["label"]

num_label = len(set(y))
labels = list(set(y))
labels2ind = {}

for index in range(len(labels)):
  labels2ind.update({labels[index]: index}) 

y = [labels2ind[a] for a in y]

X = np.array(X)
y = np.array(y)

# Suppose we have a working X, and y, both 2D

X = X.reshape((-1, sentence_length // 4, WORD_SIZE))
# y = y.reshape((-1, )

X_train = X[:int(X.shape[0] * 0.8)]
X_test = X[int(X.shape[0] * 0.8):]

y_train = y[:int(X.shape[0] * 0.8)]
y_test = y[int(X.shape[0] * 0.8):]


model = Sequential()
model.add(LSTM(100, return_sequences=True))
model.add(Flatten())
model.add(Dense(num_label, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,
          y_train, 
          epochs=EPOCH)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



