from loader import Loader
from nli import NLI
import pandas as pd

def main():

    # load data
    loader = Loader("pomt")
    data = loader.load()
    
    nli = NLI()
    predictions = nli.predict(data[['claim', 'hypothesis']])
    
    predictions_df = pd.DataFrame(predictions, columns=['entailment_prob'])
    predictions_df['claimId'] = data['claimId']

    # save predictions
    predictions_df.to_csv("result/predictions.csv", index=False)


if __name__ == "__main__":
    main()

    