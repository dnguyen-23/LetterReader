import numpy as np
import pandas as pd
import string
from matplotlib import pyplot as plt
from numpy import savetxt
data = pd.read_csv("C:/Users/danie/Downloads/emnist-letters-train.csv.zip")
data = np.array(data)
orig_data = data
print(data.shape)
# labels are 1 - 26 to represent each letter: make sure to subtract by 1 so you can index the correct probability
data = data[0:5000] # pull 2000 samples
# orig_labels = orig_data.T[0]
# orig_data = orig_data.T[1:len(orig_data)].T / 255
y_labels = data.T[0] - 1
data = data.T[1:785].T / 255
print(data.shape)
samples, pixels = data.shape
#learn rates of 0.1 works best
learn_rate = 1
def layer(inputs, neurons):
    weights = 10**-2 * np.random.randn(inputs, neurons)
    bias = 10**-2 * np.random.randn(1, neurons)
    return weights, bias

def ReLU_act(input):
    return np.maximum(input, 0)

def softmax_function(outputs_raw):
    # print(outputs_raw.shape)
    softmax_outputs = np.exp(outputs_raw) / np.sum(np.exp(outputs_raw), axis = 1, keepdims = True)
    return softmax_outputs
# Precondition: inputs are organized samples (rows) by pixels (columns)
def forward_prop(inputs, weights1, weights2, weights3, bias1, bias2, bias3):
    layer1_raw = inputs.dot(weights1) + bias1
    # print(layer1_raw.shape) # the shape here should be 1000 by 100 neurons
    layer1_act = ReLU_act(layer1_raw)
    layer2_raw = layer1_act.dot(weights2) + bias2
    # the shape here should be 1000 by 100 neurons
    layer2_act = ReLU_act(layer2_raw)
    layer3 = layer2_act.dot(weights3) + bias3
    # layer3 = np.clip(layer3, 10**-9, 1 - 10**-9) this was the main error

    softmax_probs = softmax_function(layer3)

    return layer1_act, layer2_act, softmax_probs

def crossEntropy_loss(probabilities):
    probabilities = np.clip(probabilities, 10 ** -9, 1 - 10 ** -9)
    correct_probs = probabilities[range(samples), y_labels]
    return np.sum(-np.log(correct_probs)) / samples

def dReLU(layer_act):
    return layer_act > 0

def back_prop(inputs, weights1, layer1_act, weights2, layer2_act, weights3, softmax_prob):
    softmax_prob[range(len(softmax_prob)), y_labels] -= 1
    dlayer3 = softmax_prob / samples
    dweights3 = dlayer3.T.dot(layer2_act).T
    # dweights3 = layer2_act.T.dot(dlayer3)
    dbias3 = np.sum(dlayer3, axis = 0, keepdims = True)

    # probabilities of 1 neuron correspond with the weights for that neuron
    dlayer2 = dlayer3.dot(weights3.T) * dReLU(layer2_act)
    dweights2 = dlayer2.T.dot(layer1_act).T
    # dweights2 = layer1_act.T.dot(dlayer2)
    dbias2 = np.sum(dlayer2, axis = 0, keepdims = True)

    dlayer1 = dlayer2.dot(weights2.T) * dReLU(layer1_act)
    dweights1 = dlayer1.T.dot(inputs).T
    # dweights1 = inputs.T.dot(dlayer1)
    dbias1 = np.sum(dlayer1, axis = 0, keepdims = True)
    # print(dbias1.shape)
    return dweights1, dweights2, dweights3, dbias1, dbias2, dbias3


weights1, bias1 = layer(pixels, 500)
weights2, bias2 = layer(500, 300)
weights3, bias3 = layer(300, 26)

for i in range(400):
    layer1_act, layer2_act, softmaxProbs = forward_prop(data, weights1, weights2, weights3, bias1, bias2, bias3)
    loss = crossEntropy_loss(softmaxProbs)
    accuracy = np.sum(np.argmax(softmaxProbs, axis = 1) == y_labels) / samples
    if i % 10 == 0:
        print("Iteration", i , "Accuracy:", accuracy, "Loss:", loss)

    dweights1, dweights2, dweights3, dbias1, dbias2, dbias3 = back_prop(data, weights1, layer1_act, weights2,
                                                                        layer2_act, weights3, softmaxProbs)


    weights1 -= learn_rate * dweights1
    weights2 -= learn_rate * dweights2
    weights3 -= learn_rate * dweights3
    bias1 -= learn_rate * dbias1
    bias2 -= learn_rate * dbias2
    bias3 -= learn_rate * dbias3



layer1_act, layer2_act, softmaxProbs = forward_prop(data, weights1, weights2, weights3, bias1, bias2, bias3)

def showTrainingResults():
    for i in range(200):
        print("Iteration:", i, "Actual:", string.ascii_uppercase[y_labels[i]], "Predicted:", string.ascii_uppercase[np.argmax(softmaxProbs, axis = 1)[i]])
        # if i == 12:
        image = orig_data[i]
        image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image.T, interpolation = 'nearest')
        plt.show()

def makePredictions(numPredictions):
    pData = orig_data[5001:5000+numPredictions]
    pLabels = pData.T[0] - 1
    pData = pData.T[1:785].T / 255


    layer1_act, layer2_act, softmaxProbs = forward_prop(pData, weights1, weights2, weights3, bias1, bias2, bias3)
    for i in range(numPredictions):
        print("Sample number:", i, "Actual:", string.ascii_uppercase[pLabels[i]], "Predicted:", string.ascii_uppercase[np.argmax(softmaxProbs, axis = 1)[i]])
        # if i == 12:
        image = pData[i]
        image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(image.T, interpolation = 'nearest')
        plt.show()
makePredictions(200)