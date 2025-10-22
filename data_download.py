import urllib.request
import os
from pathlib import Path

def download_wine_data():
    """Download the wine quality dataset"""
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Dataset URL
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    output_path = 'data/winequality-red.csv'
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return
    
    print("Downloading Wine Quality Dataset...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Dataset downloaded successfully to {output_path}")
        
        # Print dataset info
        import pandas as pd
        df = pd.read_csv(output_path, sep=';')
        print(f"\nDataset Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {list(df.columns)}")
        print(f"  Target range: {df['quality'].min()} - {df['quality'].max()}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(url)
        print(f"Save it as: {output_path}")

if __name__ == '__main__':
    download_wine_data()