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

## Project Structure

```
neural-networks-study/
│
├── experiment_pytorch.py          # PyTorch experiments
├── experiment_tensorflow.py       # TensorFlow experiments
├── download_data.py               # Dataset downloader
├── setup_project.py              # Project structure setup
├── app.py                        # Streamlit dashboard
├── api.py                        # FastAPI backend
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data/                         # Dataset directory
│   └── winequality-red.csv
│
├── results/                      # Experiment results
│   ├── pytorch_summary.csv
│   ├── tensorflow_summary.csv
│   └── *.json                    # Detailed results
│
├── models/                       # Saved models (optional)
└── notebooks/                    # Jupyter notebooks (optional)
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/RamuNalla/neural-networks-study.git
cd neural-networks-study
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Project Structure

```bash
python setup_project.py
```

### Step 5: Download Dataset

```bash
python download_data.py
```

## 🚀 Quick Start

### Run All Experiments

```bash
python experiment_pytorch.py        # PyTorch experiments (takes ~10-15 minutes)

python experiment_tensorflow.py     # TensorFlow experiments (takes ~10-15 minutes)
```

### Launch Visualization Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Start API Server

```bash

python -m uvicorn api:app --reload  # Terminal 1: Start API

curl http://localhost:8000/summary  # Terminal 2: Test API
```

## 🔬 Experiments

### Experiment Categories

#### 1. **Depth Experiments** (Fixed Width = 64)
- `shallow_network`: 1 hidden layer [64]
- `medium_depth`: 2 hidden layers [64, 64]
- `deep_network`: 4 hidden layers [64, 64, 64, 64]
- `very_deep`: 6 hidden layers [64, 64, 64, 64, 64, 64]

**Learn**: How depth affects learning capacity and training time

#### 2. **Width Experiments** (Fixed Depth = 2)
- `narrow_network`: [16, 16]
- `medium_width`: [64, 64]
- `wide_network`: [256, 256]
- `very_wide`: [512, 512]

**Learn**: How width affects model capacity and overfitting

#### 3. **Activation Function Experiments**
- `activation_relu`: ReLU (baseline)
- `activation_tanh`: Tanh
- `activation_sigmoid`: Sigmoid
- `activation_leaky_relu` / `activation_elu`: Advanced activations

**Learn**: Impact of non-linearity choices

#### 4. **Learning Rate Experiments**
- `lr_low`: 0.0001
- `lr_high`: 0.01
- Baseline: 0.001

**Learn**: Convergence speed vs stability trade-off

#### 5. **Regularization Experiments**
- `with_dropout`: Dropout rate = 0.3
- `with_l2`: L2 regularization = 0.01

**Learn**: Preventing overfitting techniques

## 📈 Understanding Results

### Key Metrics

#### RMSE (Root Mean Squared Error)
- **Lower is better**
- Measures average prediction error
- Same units as target (quality score)
- Typical range: 0.5 - 0.8

#### R² Score (Coefficient of Determination)
- **Higher is better** (max = 1.0)
- Proportion of variance explained
- 0.3 = explains 30% of variance
- Typical range: 0.2 - 0.4

#### MAE (Mean Absolute Error)
- **Lower is better**
- Average absolute prediction error
- More robust to outliers than RMSE

#### Training Time
- Seconds to train the model
- Consider for production deployment

### Typical Findings

#### 🔍 Depth
- **Shallow (1-2 layers)**: Fast, may underfit
- **Medium (3-4 layers)**: Best balance for this dataset
- **Deep (5+ layers)**: Slower, diminishing returns on small datasets

#### 📏 Width
- **Narrow (<50 neurons)**: Fast but limited capacity
- **Medium (50-128)**: Good starting point
- **Wide (>256)**: Risk of overfitting, slower training

#### ⚡ Learning Rate
- **Too low (0.0001)**: Slow convergence, may not reach optimum
- **Optimal (0.001)**: Good balance
- **Too high (0.01)**: Unstable, may diverge

#### 🛡️ Regularization
- **Dropout**: Effective for overfitting, slight performance cost
- **L2**: Prevents large weights, smoother models


## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### GET `/experiments`
List all completed experiments

```bash
curl http://localhost:8000/experiments
```

#### GET `/results/{experiment_name}`
Get detailed results for specific experiment

```bash
curl http://localhost:8000/results/pytorch_medium_depth
```

#### GET `/summary`
Get summary of all experiments

```bash
curl http://localhost:8000/summary
```

#### GET `/compare?experiments=exp1,exp2,exp3`
Compare multiple experiments

```bash
curl "http://localhost:8000/compare?experiments=pytorch_shallow_network,pytorch_deep_network"
```

#### GET `/best-model?metric=test_rmse`
Get best performing model

```bash
curl "http://localhost:8000/best-model?metric=test_r2"
```

#### POST `/run-experiment`
Run custom experiment

```bash
curl -X POST "http://localhost:8000/run-experiment" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_experiment",
    "framework": "pytorch",
    "hidden_dims": [128, 64],
    "learning_rate": 0.001,
    "epochs": 100
  }'
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.