from loader import Loader
from evaluater import Evaluator
from nli import NLI
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="snes", help="dataset to use")
    args = parser.parse_args()

    dataset = args.dataset
    # load data
    loader = Loader(dataset)
    data, labels = loader.load()
    
    nli = NLI()
    predictions = nli.predict(data[['claim', 'hypothesis']])
    
    predictions_df = pd.DataFrame(predictions, columns=['entailment_prob'])
    predictions_df['claimId'] = data['claimId']

    # save predictions
    predictions_df.to_csv(f"result/{dataset}_predictions.csv", index=False)

    # Evaluate
    evaluator = Evaluator(data, labels)
    f1_macro, f1_micro = evaluator.evaluate(predictions_df)
    print(f"F1 macro: {f1_macro}")
    print(f"F1 micro: {f1_micro}")

if __name__ == "__main__":
    main()
