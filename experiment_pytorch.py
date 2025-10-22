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

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:          # loop over the hidden_dims to create a layers list
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NNExperiment:
    """Handles data loading, training, and evaluation"""
    
    def __init__(self, data_path='data/winequality-red.csv'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_and_prepare_data(self):
        """Load and preprocess the wine quality dataset"""
        df = pd.read_csv(self.data_path, sep=';')
        
        # Separate features and target
        X = df.drop('quality', axis=1).values
        y = df['quality'].values.reshape(-1, 1)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)

        self.input_dim = X_train.shape[1]
        
        print(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Input features: {self.input_dim}")

    
    def train_model(self, hidden_dims, activation='relu', learning_rate=0.001,
                   batch_size=32, epochs=100, dropout_rate=0.0, l2_reg=0.0):
        """Train a model with specified architecture and hyperparameters"""

        # Create model
        model = FlexibleNN(
            self.input_dim, 
            hidden_dims, 
            activation=activation,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }

        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val)
                val_loss = criterion(val_outputs, self.y_val).item()

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['epoch_times'].append(time.time() - epoch_start)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        training_time = time.time() - start_time

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(self.X_test)
            test_loss = criterion(test_outputs, self.y_test).item()

            # Calculate metrics
            y_pred = test_outputs.cpu().numpy()
            y_true = self.y_test.cpu().numpy()

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results = {
            'framework': 'pytorch',
            'architecture': {
                'hidden_dims': hidden_dims,
                'depth': len(hidden_dims),
                'total_params': total_params,
                'trainable_params': trainable_params
            },
            'hyperparameters': {
                'activation': activation,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'dropout_rate': dropout_rate,
                'l2_reg': l2_reg
            },
            'metrics': {
                'test_mse': float(mse),
                'test_rmse': float(rmse),
                'test_mae': float(mae),
                'test_r2': float(r2),
                'final_train_loss': float(history['train_loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1])
            },
            'training': {
                'training_time': training_time,
                'avg_epoch_time': np.mean(history['epoch_times']),
                'history': history
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
        {'name': 'activation_leaky_relu', 'hidden_dims': [64, 64], 'activation': 'leaky_relu', 'learning_rate': 0.001},
        
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
        with open(f'results/pytorch_{name}.json', 'w') as f:
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

    summary.to_csv('results/pytorch_summary.csv', index=False)
    print(f"\n{'='*60}")
    print("All experiments completed! Results saved to 'results/' directory")
    print(f"{'='*60}")

    return all_results

if __name__ == '__main__':
    results = run_experiments()