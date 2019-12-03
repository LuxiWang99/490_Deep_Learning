# CSE 490G1 Deep Learning

Project Files for CSE 490G Final Project

Cailin Winston, Luxi Wang, Mitchell Estberg

cailinw, louis99, estbergm @ uw (dot) edu

Any .midi or .wav files are in /data

# Project Proposal Questions

What is the problem you are trying to solve?
Why is this a cool problem to work on?
What data do you plan to use? Do you need to gather and label it or does the data already exist?
What approaches are you planning to try?
How can you make your project 10% cooler with 1% extra work? (Hint: live demos).

# Project Proposal Response

We want to make a classifier for classical music. We will explore classifying for [music period](https://www.naxos.com/education/brief_history.asp) (Baroque, Classical, Romantic, Modern, etc.) and perhaps for composers as well. It is an interesting classification problem as there is variation amongst these styles that with a traditional model or classification program would require human feature selection from domain experts. There are even many humans that struggle with this classification task and upon hearing a new piece may have a hard time identifying the period it originated from, it takes years of music and theory study to classify these periods accurately. However, we hope to train a neural net that can handle this classification problem. 

There are hundreds of classical music .midi ([Musical Instrument Digital Interface](https://en.wikipedia.org/wiki/MIDI)) files available on the internet from various sources ([piano midi](http://www.piano-midi.de/), [maestro](https://magenta.tensorflow.org/datasets/maestro), [musedata](https://musedata.org/)) that we plan to use as data. Most of these are sorted by composer which we will have to then write a labeling script to derive period from composer. We will have to carefully handle some edge cases of composers that wrote accross multiple periods. We intend to convert the .midi files to .wav ([Waveform Audio File Format](https://en.wikipedia.org/wiki/WAV)) to a [Mel-frequency cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFC) which can be classified as an image classification problem with the low-number of classes of classical music periods, or the higer-number of classes and more challenging problem of composers. 

We would love to try a live-demo where we record some samples of performances and test our model on these recordings!

# Set Up

1. First clone this repo

`git clone https://github.com/LuxiWang99/490_Deep_Learning.git`

`cd 490_Deep_Learning`

2. Next set up the environment. The environments name will be `testenv`

`conda env create -f environment.yml`

`conda activate testenv`

3. Now, run our lab.py to download our data and run main

`python src/lab.py`
