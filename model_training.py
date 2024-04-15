import pandas as pd
import torch
from torch.utils.data import Dataset

class RedditPostDataset(Dataset):
    def __init__(self, data):
        self.text = data['text']
        self.input_id = data['input_id']
        self.attention_mask = data['attention_mask']
        self.label = data['label']
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_id[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.label[idx])
        }