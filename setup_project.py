"""
Setup script to create project structure
"""

from pathlib import Path

def create_project_structure():
    """Create the directory structure for the project"""
    
    directories = [
        'data',
        'results',
        'models',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}/")
    
    # Create .gitignore
    gitignore_content = """# Data
data/
*.csv

# Results
results/
*.json

# Models
models/
*.pth
*.h5
*.pkl

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Run: python download_data.py")
    print("2. Run: python experiment_pytorch.py")
    print("3. Run: python experiment_tensorflow.py")
    print("4. Run: streamlit run app.py")

if __name__ == '__main__':
    create_project_structure()