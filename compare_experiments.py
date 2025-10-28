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

