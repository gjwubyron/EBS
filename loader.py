import pandas as pd
import numpy as np
import pickle

class Loader:
    def __init__(self, data_path):
        base_path = "multi_fc_publicdata/"
        self.labels = pickle.load(open(f"{base_path}{data_path}/{data_path}_labels.pkl", "rb"))
        data = pd.read_csv(f"{base_path}{data_path}/{data_path}.tsv",
                                     sep='\t', header=None)
        index_splits = pickle.load(open(f"{base_path}{data_path}/{data_path}_index_split.pkl", "rb"))
        test_index = index_splits[2]
        self.test_data = data.iloc[test_index]
        # only keep the first 3 columns and column 6
        self.test_data = self.test_data.iloc[:, [0, 1, 2, 6]]
        # rename columns
        self.test_data.columns = ['claimId', 'claim', 'label', 'speaker']

    def generate_hypotheses(self):
        hypotheses = []
        print(self.labels)
        for label in self.labels:
            hypotheses.append(f"This claim is {label}")
        return hypotheses

    def load(self):
        # for each claim in test_data, merge it with all hypotheses
        hypotheses = self.generate_hypotheses()
        data = []
        for _, row in self.test_data.iterrows():
            for h in hypotheses:
                data.append([row['claimId'], row['claim'], h, row['speaker']])
        data = pd.DataFrame(data, columns=['claimId', 'claim', 'hypothesis', 'speaker'])
        return data
        