# Digit Classifier
Contains a few different neural networks trained on the MNIST dataset to classify hand written digits. The best performing network was a convolutional network with 3 hidden layers which had an accuracy of 98% on the testing dataset

Since the network performed so well I decided to make an API so that I can develop a small application around the model.

## Project Tree
```bash
├───backend
│   ├───templates
└───model
    ├───notebooks
    ├───outputs
    └───src
```

### Original README: MNIST Simple Neural Network
---

A simple neural network to classify handwritten digit, made with PyTorch and trained on a subset of the MNIST dataset.

The network itself is pretty simple. It is made up of an input layer with 784 nodes (28x28) image, two hidden linear layers with 14 nodes each (chosen arbitrarily), and an output layer with 10 nodes  (digits one through nine).

Although this is a simple implementation of a neural network, I hope that coding it myself will help bridge the gap between my understanding of the concept and its implementation.

This model is trained on a subset of the MNIST dataset found [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
