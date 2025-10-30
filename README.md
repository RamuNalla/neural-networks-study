# Neural Network Architecture Explorer

A comprehensive educational project for experimenting with neural network depth, width, and hyperparameters using both PyTorch and TensorFlow. Perfect for understanding how different architectural choices impact model performance on tabular data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

##  Overview

This project uses the **Wine Quality Dataset** (physicochemical properties → quality prediction) to systematically explore:

- **Network Depth**: Impact of adding more layers
- **Network Width**: Impact of more neurons per layer
- **Activation Functions**: ReLU, Tanh, Sigmoid, ELU/LeakyReLU
- **Learning Rate**: Effect on convergence and performance
- **Regularization**: Dropout and L2 regularization
- **Framework Comparison**: PyTorch vs TensorFlow implementations