import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, data, labels, dataset):
        self.ground_truth = data[['claimId', 'label']]
        self.labels = labels
        self.dataset = dataset

    def assign_label(self, predictions_df):
        predictions = []
        for claimId, group in predictions_df.groupby('claimId'):
            max_index = np.argmax(group['entailment_prob'])
            predictions.append([claimId, self.labels[max_index]])
        predictions = pd.DataFrame(predictions, columns=['claimId', 'label_pred'])
        # save predictions
        predictions.to_csv(f"result/{self.dataset}_labels.csv", index=False)
        return predictions

    def evaluate(self, predictions_df):
        predictions = self.assign_label(predictions_df)
        merged = pd.merge(predictions, self.ground_truth, on='claimId')
        merged['label'] = merged['label'].apply(lambda x: self.labels.index(x))
        merged['label_pred'] = merged['label_pred'].apply(lambda x: self.labels.index(x))
        f1_maro = f1_score(merged['label'], merged['label_pred'], average='macro')
        f1_micro = f1_score(merged['label'], merged['label_pred'], average='micro')
        return f1_maro, f1_micro
