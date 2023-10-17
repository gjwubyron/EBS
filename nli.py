# 0 entailment 1 neutral 2 contradiction
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class NLI:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

    def load(self, data):
        encoded_data = self.tokenizer.batch_encode_plus(
            data.values.tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=32)
        return data_loader

    def predict(self, data):
        data_loader = self.load(data)
        predictions = []
        self.model.eval()

        for batch in data_loader:
            batch = tuple(b.to(self.device) for b in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            entail_contradiction_logits = logits[:,[0,2]]
            probs = torch.nn.functional.softmax(torch.tensor(entail_contradiction_logits), dim=1)
            # store prob of entailment
            predictions.append(probs[:,0])
            
        predictions = torch.cat(predictions)
        return predictions.numpy()

