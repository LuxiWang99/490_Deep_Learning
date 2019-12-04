import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot(x_values, y_values, title, xlabel, ylabel):
    """Plots a line graph

    Args:
        x_values(list or np.array): x values for the line
        y_values(list or np.array): y values for the line
        title(str): Title for the plot
        xlabel(str): Label for the x axis
        ylabel(str): label for the y axis
    """

    fig = plt.figure(num=title, figsize=(20, 10))
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(title + '.png', dpi=fig.dpi)

infile = open('result.pickle', 'rb')
results = pickle.load(infile, encoding='bytes')

epochs = np.arange(10) + 1

train_acc = results['train_acc']
train_loss = results['train_loss']
test_acc = results['test_acc']

plot(epochs, train_acc, 'TrainAccuracy', 'epoch', 'Training Accuracy')
plot(epochs, train_loss, 'TrainLoss', 'epoch', 'Training Loss')
plot(epochs, test_acc, 'TestAccuracy', 'epoch', 'Test Accuracy')
