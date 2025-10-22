import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import time
from pathlib import Path

class FlexibleNN(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim=1, 
                 activation='relu', dropout_rate=0.0):
        super(FlexibleNN, self).__init__()

        # Choose activation function
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_functions.get(activation, nn.ReLU())