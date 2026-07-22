import sys
import os

# Ensure the local packages can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_rpkclust import run_all_experiments

if __name__ == "__main__":
    print("Initializing RPKClust Project Environment...")
    run_all_experiments()