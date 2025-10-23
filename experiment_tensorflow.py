import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import time
from pathlib import Path

class FlexibleNN:
    """Flexible Neural Network builder with configurable depth and width"""

    @staticmethod
    def build_model(input_dim, hidden_dims, output_dim=1, 
                   activation='relu', dropout_rate=0.0, l2_reg=0.0):

