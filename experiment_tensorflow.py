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

        # Evaluate on test set
        test_loss, test_mae = model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Predictions
        y_pred = model.predict(self.X_test, verbose=0).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

        # Prepare results
        results = {
            'framework': 'tensorflow',
            'architecture': {
                'hidden_dims': hidden_dims,
                'depth': len(hidden_dims),
                'total_params': int(total_params),
                'trainable_params': int(trainable_params)
            },
            'hyperparameters': {
                'activation': activation,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'actual_epochs': len(history.history['loss']),
                'dropout_rate': dropout_rate,
                'l2_reg': l2_reg
            },
            'metrics': {
                'test_mse': float(mse),
                'test_rmse': float(rmse),
                'test_mae': float(mae),
                'test_r2': float(r2),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            },
            'training': {
                'training_time': training_time,
                'avg_epoch_time': training_time / len(history.history['loss']),
                'history': {
                    'train_loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'train_mae': [float(x) for x in history.history['mae']],
                    'val_mae': [float(x) for x in history.history['val_mae']]
                }
            }
        }
        
        return model, results
    
def run_experiments():
    """Run a series of experiments with different configurations"""

    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Initialize experiment
    exp = NNExperiment()
    exp.load_and_prepare_data()

    # Define experiments
    experiments = [
        # Depth experiments (fixed width)
        {'name': 'shallow_network', 'hidden_dims': [64], 'learning_rate': 0.001},
        {'name': 'medium_depth', 'hidden_dims': [64, 64], 'learning_rate': 0.001},
        {'name': 'deep_network', 'hidden_dims': [64, 64, 64, 64], 'learning_rate': 0.001},
        {'name': 'very_deep', 'hidden_dims': [64, 64, 64, 64, 64, 64], 'learning_rate': 0.001},
        
        # Width experiments (fixed depth=2)
        {'name': 'narrow_network', 'hidden_dims': [16, 16], 'learning_rate': 0.001},
        {'name': 'medium_width', 'hidden_dims': [64, 64], 'learning_rate': 0.001},
        {'name': 'wide_network', 'hidden_dims': [256, 256], 'learning_rate': 0.001},
        {'name': 'very_wide', 'hidden_dims': [512, 512], 'learning_rate': 0.001},
        
        # Activation function experiments
        {'name': 'activation_tanh', 'hidden_dims': [64, 64], 'activation': 'tanh', 'learning_rate': 0.001},
        {'name': 'activation_sigmoid', 'hidden_dims': [64, 64], 'activation': 'sigmoid', 'learning_rate': 0.001},
        {'name': 'activation_elu', 'hidden_dims': [64, 64], 'activation': 'elu', 'learning_rate': 0.001},
        
        # Learning rate experiments
        {'name': 'lr_low', 'hidden_dims': [64, 64], 'learning_rate': 0.0001},
        {'name': 'lr_high', 'hidden_dims': [64, 64], 'learning_rate': 0.01},
        
        # Regularization experiments
        {'name': 'with_dropout', 'hidden_dims': [128, 128], 'dropout_rate': 0.3, 'learning_rate': 0.001},
        {'name': 'with_l2', 'hidden_dims': [128, 128], 'l2_reg': 0.01, 'learning_rate': 0.001},
    ]
    
    all_results = []

    for i, config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {config['name']}")
        print(f"{'='*60}")
        
        # Extract config
        name = config.pop('name')
        
        # Set defaults
        config.setdefault('activation', 'relu')
        config.setdefault('batch_size', 32)
        config.setdefault('epochs', 100)
        config.setdefault('dropout_rate', 0.0)
        config.setdefault('l2_reg', 0.0)
        
        # Train model
        model, results = exp.train_model(**config)
        results['experiment_name'] = name
        
        # Save results
        with open(f'results/tensorflow_{name}.json', 'w') as f:
            # Remove history for cleaner JSON
            results_to_save = results.copy()
            results_to_save['training'].pop('history')
            json.dump(results_to_save, f, indent=2)
        
        all_results.append(results)
        
        print(f"\nResults for {name}:")
        print(f"  Test RMSE: {results['metrics']['test_rmse']:.4f}")
        print(f"  Test R2: {results['metrics']['test_r2']:.4f}")
        print(f"  Parameters: {results['architecture']['total_params']}")
        print(f"  Training time: {results['training']['training_time']:.2f}s")
        
        # Clear session to free memory
        keras.backend.clear_session()

    # Save summary
    summary = pd.DataFrame([
        {
            'experiment': r['experiment_name'],
            'framework': r['framework'],
            'depth': r['architecture']['depth'],
            'hidden_dims': str(r['architecture']['hidden_dims']),
            'total_params': r['architecture']['total_params'],
            'activation': r['hyperparameters']['activation'],
            'learning_rate': r['hyperparameters']['learning_rate'],
            'dropout': r['hyperparameters']['dropout_rate'],
            'l2_reg': r['hyperparameters']['l2_reg'],
            'test_rmse': r['metrics']['test_rmse'],
            'test_mae': r['metrics']['test_mae'],
            'test_r2': r['metrics']['test_r2'],
            'training_time': r['training']['training_time']
        }
        for r in all_results
    ])

    summary.to_csv('results/tensorflow_summary.csv', index=False)
    print(f"\n{'='*60}")
    print("All experiments completed! Results saved to 'results/' directory")
    print(f"{'='*60}")
    
    return all_results

if __name__ == '__main__':
    results = run_experiments()

