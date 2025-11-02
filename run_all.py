import subprocess
import sys
from pathlib import Path
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_command(command, description):
    """Run a command and track time"""
    print(f"▶ {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✓ {description} completed in {elapsed:.1f}s\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        print(f"Error: {e}\n")
        return False
    
def main():
    """Main execution function"""
    
    print_header("🧠 Neural Network Architecture Explorer")
    print("This script will:")
    print("  1. Setup project structure")
    print("  2. Download dataset")
    print("  3. Run PyTorch experiments")
    print("  4. Run TensorFlow experiments")
    print("  5. Launch visualization dashboard")
    print("\nEstimated time: 20-30 minutes\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Setup
    print_header("Step 1: Project Setup")
    if not run_command(f"{sys.executable} setup_project.py", "Setting up project structure"):
        return
    
    # Step 2: Download data
    print_header("Step 2: Download Dataset")
    if not run_command(f"{sys.executable} data_download.py", "Downloading Wine Quality dataset"):
        print("⚠ Warning: Dataset download failed. Please download manually.")
        return
    
    # Step 3: PyTorch experiments
    print_header("Step 3: PyTorch Experiments")
    print("Running 15 experiments with different configurations...")
    if not run_command(f"{sys.executable} experiment_pytorch.py", "PyTorch experiments"):       # Test changes
        print("⚠ Warning: PyTorch experiments failed.")
    
    # Step 4: TensorFlow experiments
    print_header("Step 4: TensorFlow Experiments")
    print("Running 15 experiments with different configurations...")
    if not run_command(f"{sys.executable} experiment_tensorflow.py", "TensorFlow experiments"):
        print("⚠ Warning: TensorFlow experiments failed.")
    
    # Summary
    print_header("✅ Experiments Complete!")
    
    # Check results
    results_dir = Path('results')
    if results_dir.exists():
        json_files = list(results_dir.glob('*.json'))
        csv_files = list(results_dir.glob('*.csv'))
        print(f"Generated {len(json_files)} experiment results")
        print(f"Generated {len(csv_files)} summary files")
    
    print("\n" + "-"*60)
    print("Next Steps:")
    print("-"*60)
    print("\n1. Launch Streamlit Dashboard:")
    print("   streamlit run app.py")
    print("\n2. Start FastAPI Server:")
    print("   python -m uvicorn api:app --reload")
    print("\n3. View results directory:")           # test
    print("   ls results/")
    print("\n4. Explore detailed results:")
    print("   python")
    print("   >>> import json")
    print("   >>> with open('results/pytorch_medium_depth.json') as f:")
    print("   >>>     data = json.load(f)")
    print("   >>>     print(data['metrics'])")
    
    print("\n" + "="*60)
    print("  Happy Experimenting! 🚀")
    print("="*60 + "\n")
    
    # Ask if user wants to launch Streamlit
    launch = input("\nLaunch Streamlit dashboard now? (y/n): ")
    if launch.lower() == 'y':
        print("\nLaunching Streamlit...")
        print("(Press Ctrl+C to stop)\n")
        subprocess.run("streamlit run app.py", shell=True)

if __name__ == '__main__':
    main()