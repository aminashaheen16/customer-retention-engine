import subprocess
import sys
import os

if not os.path.exists("models/churn_model.pkl"):
    print("Training models...")
    subprocess.run([sys.executable, "main.py"], check=True)
