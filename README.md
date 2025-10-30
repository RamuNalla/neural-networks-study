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