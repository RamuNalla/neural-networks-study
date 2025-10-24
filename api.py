from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import pandas as pd
from pathlib import Path
import asyncio
from datetime import datetime

app = FastAPI(title="NN Architecture Explorer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ExperimentConfig(BaseModel):
    name: str
    framework: str  # 'pytorch' or 'tensorflow'
    hidden_dims: List[int]
    activation: str = 'relu'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.0
    l2_reg: float = 0.0

class ExperimentStatus(BaseModel):
    status: str
    message: str

# Global state for tracking experiments
experiment_status = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neural Network Architecture Explorer API",
        "version": "1.0.0",
        "endpoints": {
            "experiments": "/experiments",
            "results": "/results",
            "summary": "/summary",
            "compare": "/compare"
        }
    }

@app.get("/experiments")
async def list_experiments():
    """List all completed experiments"""
    results_dir = Path('results')
    if not results_dir.exists():
        return {"experiments": []}
    
    experiments = []
    for file in results_dir.glob('*.json'):
        experiments.append(file.stem)
    
    return {
        "count": len(experiments),
        "experiments": sorted(experiments)
    }

@app.get("/results/{experiment_name}")
async def get_experiment_results(experiment_name: str):
    """Get detailed results for a specific experiment"""
    result_file = Path(f'results/{experiment_name}.json')
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return results

@app.get("/summary")
async def get_summary():
    """Get summary of all experiments"""
    pytorch_summary = Path('results/pytorch_summary.csv')
    tensorflow_summary = Path('results/tensorflow_summary.csv')
    
    summaries = {}
    
    if pytorch_summary.exists():
        df = pd.read_csv(pytorch_summary)
        summaries['pytorch'] = df.to_dict(orient='records')
    
    if tensorflow_summary.exists():
        df = pd.read_csv(tensorflow_summary)
        summaries['tensorflow'] = df.to_dict(orient='records')
    
    if not summaries:
        raise HTTPException(status_code=404, detail="No experiment summaries found. Run experiments first.")
    
    return summaries


@app.get("/compare")
async def compare_experiments(experiments: str):
    """Compare multiple experiments
    
    Args:
        experiments: Comma-separated list of experiment names
    """
    exp_list = [e.strip() for e in experiments.split(',')]
    
    results = []
    for exp_name in exp_list:
        result_file = Path(f'results/{exp_name}.json')
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append(data)
    
    if not results:
        raise HTTPException(status_code=404, detail="No matching experiments found")
    
    # Create comparison
    comparison = {
        'experiments': [r.get('experiment_name', 'unknown') for r in results],
        'frameworks': [r.get('framework', 'unknown') for r in results],
        'metrics': {
            'test_rmse': [r['metrics']['test_rmse'] for r in results],
            'test_r2': [r['metrics']['test_r2'] for r in results],
            'test_mae': [r['metrics']['test_mae'] for r in results]
        },
        'parameters': [r['architecture']['total_params'] for r in results],
        'training_time': [r['training']['training_time'] for r in results]
    }
    
    return comparison

@app.post("/run-experiment")
async def run_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """Run a custom experiment with specified configuration"""
    
    experiment_id = f"{config.framework}_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Update status
    experiment_status[experiment_id] = {
        "status": "queued",
        "message": "Experiment queued for execution"
    }
    
    # Add to background tasks
    background_tasks.add_task(execute_experiment, experiment_id, config)
    
    return {
        "experiment_id": experiment_id,
        "status": "queued",
        "message": "Experiment started in background"
    }

@app.get("/experiment-status/{experiment_id}")
async def get_experiment_status(experiment_id: str):
    """Get the status of a running experiment"""
    if experiment_id not in experiment_status:
        raise HTTPException(status_code=404, detail="Experiment ID not found")
    
    return experiment_status[experiment_id]

async def execute_experiment(experiment_id: str, config: ExperimentConfig):
    """Execute an experiment (runs in background)"""
    
    try:
        experiment_status[experiment_id] = {
            "status": "running",
            "message": "Experiment in progress"
        }
        
        # Import appropriate module
        if config.framework == 'pytorch':
            import experiment_pytorch as exp_module
        elif config.framework == 'tensorflow':
            import experiment_tensorflow as exp_module
        else:
            raise ValueError(f"Unknown framework: {config.framework}")
        
        # Load data
        exp = exp_module.NNExperiment()
        exp.load_and_prepare_data()
        
        # Train model
        model, results = exp.train_model(
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            dropout_rate=config.dropout_rate,
            l2_reg=config.l2_reg
        )
        
        results['experiment_name'] = config.name
        
        # Save results
        output_file = f'results/{experiment_id}.json'
        with open(output_file, 'w') as f:
            results_to_save = results.copy()
            if 'history' in results_to_save['training']:
                results_to_save['training'].pop('history')
            json.dump(results_to_save, f, indent=2)
        
        experiment_status[experiment_id] = {
            "status": "completed",
            "message": "Experiment completed successfully",
            "results_file": output_file
        }
        
    except Exception as e:
        experiment_status[experiment_id] = {
            "status": "failed",
            "message": f"Experiment failed: {str(e)}"
        }

@app.get("/best-model")
async def get_best_model(metric: str = 'test_rmse'):
    """Get the best performing model based on a specific metric"""
    
    valid_metrics = ['test_rmse', 'test_mae', 'test_r2', 'training_time']
    if metric not in valid_metrics:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid metric. Choose from: {valid_metrics}"
        )
    
    # Load all summaries
    all_results = []
    
    for framework in ['pytorch', 'tensorflow']:
        summary_file = Path(f'results/{framework}_summary.csv')
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            all_results.extend(df.to_dict(orient='records'))
    
    if not all_results:
        raise HTTPException(status_code=404, detail="No experiments found")
    
    # Find best model
    if metric == 'test_r2':
        best = max(all_results, key=lambda x: x[metric])
    else:
        best = min(all_results, key=lambda x: x[metric])
    
    return {
        "metric": metric,
        "best_experiment": best['experiment'],
        "framework": best['framework'],
        "value": best[metric],
        "details": best
    }


