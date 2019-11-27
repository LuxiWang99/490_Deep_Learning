#!/usr/bin/env python
# coding: utf-8

# In[13]:


import keras
import numpy as np
import librosa
import sys

class SpectogramDataLoader(keras.utils.Sequence):

    '''
    song_IDs : np.ndarray
        List of song IDs in the data
    labels : dict
        Dictionary mapping each song ID to time period label
    data_path : string
        Path of the .npy data files
    batch_size : int
        Batch size
    hop_length : int (default 512)
        Parameter for librosa.stft
    win_length : int (default 2048)
        Parameter for librosa.stft
    SR : int (default 44100)
        Sample rate; number of values per second
    '''
    def __init__(self, song_IDs, labels, data_path, batch_size, hop_length=512,
                 win_length=2048, SR=44100, shuffle=True):
        self.song_IDs = song_IDs
        self.labels = labels
        self.data_path = data_path
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.SR = SR
        self.shuffle = shuffle
        self.labels2ind = list(set(labels.values()))
        self.set_data_dim()
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
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(song_IDs_temp):
            # Store sample
            x_ = np.load(self.data_path + str(ID) + '.npy')
            X[i,] = np.log((np.abs(librosa.stft(x_, hop_length=2048, win_length=512))**2) + sys.float_info.epsilon)

            # Store class
            y[i] = self.labels[ID]
        
        # Categorical to numerical labels
        y = [self.labels2ind.index(a) for a in y]

        return X, y
    
    def set_data_dim(self):
        'Sets the dimension of one data sample'
        x_ = np.load(self.data_path + str(self.song_IDs[0]) + '.npy')
        D = np.abs(librosa.stft(x_, hop_length=self.hop_length, win_length=self.win_length))**2
        self.data_dim = D.shape
    
    def get_data_dim(self):
        'Returns dimension of one data sample'
        return self.data_dim
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.song_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.song_IDs)

