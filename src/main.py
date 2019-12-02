from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
import pickle 
import os
import librosa
from spectogram_dataloader import SpectogramDataLoader


LENGTH=3 * 44100
def get_model():
    model = Sequential()
    model.add(LSTM(1025, input_shape=(1025, 259), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(4, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(model, train_generator, EPOCHS=20):
    model.fit_generator(generator=train_generator, epochs=EPOCHS)

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
            print("right! prediction: " + str(predict_label) + ", label: " + str(label))
        else:
            print("wrong! prediction: " + str(predict_label) + ", label: " + str(label))
    return float(num_corr) / len(X_test)

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


pwd = os.getcwd()
data_path = pwd + os.sep + 'data/'
train_song_ids_file =  pwd + os.sep + 'data/train/song_ids.pkl'
test_song_ids_file =  pwd + os.sep +'data/test/song_ids.pkl'
labels_file =  pwd + os.sep +'data/labels.pkl'

with open(train_song_ids_file, 'rb') as handle:
    train_ids = pickle.load(handle)

with open(test_song_ids_file, 'rb') as handle:
    test_ids = pickle.load(handle)

with open(labels_file, 'rb') as handle:
    labels = pickle.load(handle)



train_generator = SpectogramDataLoader(train_ids, labels, data_path + os.sep + 'train/', 
batch_size=64)
print(train_generator.get_data_dim())
X_test, y_test = get_test_data(test_dir=data_path + os.sep + 'test/', labels_dict=labels)
model = get_model()
train(model, train_generator)
# acc = test(model, X_test, y_test)

# print("Accuracy: %.2f%%" % (acc*100))

# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))



# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# from keras.models import model_from_json

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights("model.h5")
# print("Loaded model from disk")