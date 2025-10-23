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

        """Build a Keras model with specified architecture"""

        model = models.Sequential()

        # Input layer
        model.add(layers.InputLayer(input_shape=(input_dim,)))

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            if l2_reg > 0:
                model.add(layers.Dense(
                    hidden_dim, 
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=f'hidden_{i+1}'
                ))
            else:
                model.add(layers.Dense(
                    hidden_dim, 
                    activation=activation,
                    name=f'hidden_{i+1}'
                ))

            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

        # Output layer
        model.add(layers.Dense(output_dim, name='output'))

        return model
    
class NNExperiment:
    """Handles data loading, training, and evaluation"""

    def __init__(self, data_path='data/winequality-red.csv'):
        self.data_path = data_path
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                print(f"  {gpu}")
        else:
            print("Running on CPU")

    def load_and_prepare_data(self):
        """Load and preprocess the wine quality dataset"""
        # Load data
        df = pd.read_csv(self.data_path, sep=';')
        
        # Separate features and target
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.input_dim = self.X_train.shape[1]
        
        print(f"Data loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        print(f"Input features: {self.input_dim}")

    def train_model(self, hidden_dims, activation='relu', learning_rate=0.001,
                   batch_size=32, epochs=100, dropout_rate=0.0, l2_reg=0.0):
        """Train a model with specified architecture and hyperparameters"""
        
        # Build model
        model = FlexibleNN.build_model(
            self.input_dim,
            hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )

        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        # Training callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )

        # Train model
        start_time = time.time()
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        training_time = time.time() - start_time


