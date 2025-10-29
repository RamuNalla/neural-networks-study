import sys
import json
from pathlib import Path
import pandas as pd

def load_experiment(name):
    """Load experiment results"""
    result_file = Path(f'results/{name}.json')
    if not result_file.exists():
        return None
    
    with open(result_file, 'r') as f:
        return json.load(f)

def compare_experiments(experiment_names):
    """Compare multiple experiments and display results"""
    
    results = []
    missing = []
    
    for name in experiment_names:
        exp_data = load_experiment(name)
        if exp_data is None:
            missing.append(name)
        else:
            results.append(exp_data)
    
    if missing:
        print(f"⚠ Warning: Could not find experiments: {', '.join(missing)}\n")
    
    if not results:
        print("✗ No valid experiments found!")
        return
    
    print("\n" + "="*80)
    print("  📊 EXPERIMENT COMPARISON")
    print("="*80 + "\n")
    
    # Create comparison table
    comparison_data = []
    
    for exp in results:
        row = {
            'Experiment': exp.get('experiment_name', 'Unknown'),
            'Framework': exp.get('framework', 'Unknown'),
            'Architecture': str(exp['architecture']['hidden_dims']),
            'Depth': exp['architecture']['depth'],
            'Parameters': f"{exp['architecture']['total_params']:,}",
            'Activation': exp['hyperparameters']['activation'],
            'Learning Rate': exp['hyperparameters']['learning_rate'],
            'Dropout': exp['hyperparameters']['dropout_rate'],
            'L2 Reg': exp['hyperparameters']['l2_reg'],
            'Test RMSE': f"{exp['metrics']['test_rmse']:.4f}",
            'Test R²': f"{exp['metrics']['test_r2']:.4f}",
            'Test MAE': f"{exp['metrics']['test_mae']:.4f}",
            'Train Time (s)': f"{exp['training']['training_time']:.2f}"
        }
        comparison_data.append(row)
    
    # Display as DataFrame
    df = pd.DataFrame(comparison_data)
    
    print("Architecture Details:")
    print("-" * 80)
    print(df[['Experiment', 'Framework', 'Architecture', 'Depth', 'Parameters']].to_string(index=False))
    
    print("\n\nHyperparameters:")
    print("-" * 80)
    print(df[['Experiment', 'Activation', 'Learning Rate', 'Dropout', 'L2 Reg']].to_string(index=False))
    
    print("\n\nPerformance Metrics:")
    print("-" * 80)
    print(df[['Experiment', 'Test RMSE', 'Test R²', 'Test MAE', 'Train Time (s)']].to_string(index=False))
    
    # Find best
    print("\n\n" + "="*80)
    print("  🏆 WINNERS")
    print("="*80 + "\n")
    
    rmse_values = [float(r['metrics']['test_rmse']) for r in results]
    r2_values = [float(r['metrics']['test_r2']) for r in results]
    time_values = [float(r['training']['training_time']) for r in results]
    
    best_rmse_idx = rmse_values.index(min(rmse_values))
    best_r2_idx = r2_values.index(max(r2_values))
    fastest_idx = time_values.index(min(time_values))
    
    print(f"🥇 Best RMSE: {results[best_rmse_idx]['experiment_name']} "
          f"({rmse_values[best_rmse_idx]:.4f})")
    print(f"🥇 Best R²: {results[best_r2_idx]['experiment_name']} "
          f"({r2_values[best_r2_idx]:.4f})")
    print(f"⚡ Fastest: {results[fastest_idx]['experiment_name']} "
          f"({time_values[fastest_idx]:.2f}s)")
    
    # Performance insights
    print("\n\n" + "="*80)
    print("  💡 INSIGHTS")
    print("="*80 + "\n")
    
    # RMSE improvement
    rmse_improvement = (max(rmse_values) - min(rmse_values)) / max(rmse_values) * 100
    print(f"✓ RMSE improvement: {rmse_improvement:.1f}% "
          f"(from {max(rmse_values):.4f} to {min(rmse_values):.4f})")
    
    # R² improvement
    r2_improvement = (max(r2_values) - min(r2_values)) / min(r2_values) * 100
    print(f"✓ R² improvement: {r2_improvement:.1f}% "
          f"(from {min(r2_values):.4f} to {max(r2_values):.4f})")
    
    # Time comparison
    time_diff = max(time_values) - min(time_values)
    print(f"✓ Time difference: {time_diff:.1f}s "
          f"({max(time_values)/min(time_values):.1f}x slower)")
    
    # Parameter count
    param_counts = [r['architecture']['total_params'] for r in results]
    print(f"✓ Parameter range: {min(param_counts):,} to {max(param_counts):,}")
    
    # Framework comparison if applicable
    frameworks = [r['framework'] for r in results]
    if len(set(frameworks)) > 1:
        print("\n" + "Framework Comparison:")
        for fw in set(frameworks):
            fw_results = [r for r in results if r['framework'] == fw]
            fw_avg_rmse = sum([r['metrics']['test_rmse'] for r in fw_results]) / len(fw_results)
            fw_avg_time = sum([r['training']['training_time'] for r in fw_results]) / len(fw_results)
            print(f"  {fw.upper()}: Avg RMSE = {fw_avg_rmse:.4f}, Avg Time = {fw_avg_time:.2f}s")
    
    print("\n" + "="*80 + "\n")