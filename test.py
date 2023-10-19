import pandas as pd
from evaluater import Evaluator
import pickle
from loader import Loader

# Load data
dataset = "pomt"
loader = Loader(dataset)
data, labels = loader.load()

# Load predictions
predictions_df = pd.read_csv(f"result/{dataset}_predictions.csv")

# Evaluate
evaluator = Evaluator(data, labels)
f1_macro, f1_micro = evaluator.evaluate(predictions_df)
print(f"F1 macro: {f1_macro}")
print(f"F1 micro: {f1_micro}")
