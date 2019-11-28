#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM

class SentenceDataLoader(keras.utils.Sequence):

    '''
    song_IDs : np.ndarray
        List of song IDs in the data
    labels : dict
        Dictionary mapping each song ID to time period label
    data_path : string
        Path of the .npy data files
    batch_size : int
        Batch size
    sentence_length : int (default 30)
        Number of seconds each data sample (.npy file) represents
    word_length : int (default 4)
        Number of seconds each word represents; each sentence is split into words of this length
    SR : int (default 44100)
        Sample rate; number of values per second
    '''
    def __init__(self, song_IDs, labels, data_path, batch_size, sentence_length=30,
                 word_length=5, SR=44100, shuffle=True):
        self.song_IDs = song_IDs
        self.labels = labels
        self.data_path = data_path
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.word_length = word_length
        self.SR = SR
        self.data_dim = (sentence_length // word_length, word_length * SR)
        self.shuffle = shuffle
        self.labels2ind = ["Bar", "Cla", "Rom", "Mod"]
        self.on_epoch_end()
        
    def __len__(self):
        'Returns the number of batches per epoch'
        return int(np.floor(len(self.song_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        song_IDs_temp = [self.song_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(song_IDs_temp)

        return X, y
    
    def __data_generation(self, song_IDs_temp):
        'Generates data containing batch_size samples' # X : (batch_size, num_words, word_length)
        # Initialization
        X = np.empty((self.batch_size, *self.data_dim))
        y = np.empty((self.batch_size), dtype=object)

        # Generate data
        for i, ID in enumerate(song_IDs_temp):
            # Store sample
            X[i,] = (np.load(self.data_path + str(ID) + '.npy')).reshape((self.data_dim))

            # Store class
            y[i] = self.labels[ID]
        
        # Categorical to numerical labels
        y = [self.labels2ind.index(a) for a in y]

        return X, y
    
    '''
    Returns the data dimension
    '''
    def data_dim(self):
        return data_dim 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.song_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.song_IDs)

