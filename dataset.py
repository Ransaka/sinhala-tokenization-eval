import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class SinhalaHateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode(text)
        return {
            'text': text,
            'input_ids': torch.tensor(encoding),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
    @staticmethod
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        
        # Padding
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        attention_mask = []
        
        for ids in input_ids:
            padding_length = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones(len(ids), dtype=torch.long), 
                                            torch.zeros(padding_length, dtype=torch.long)]))
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': labels
        }

def load_sinhala_hate_speech_data(data_path, test_size=0.2, random_state=42):
    """
    Load and prepare Sinhala hate speech data
    
    Args:
        data_path: Path to the CSV file with Sinhala text and hate speech labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    # Load data from CSV
    df = pd.read_csv(data_path)
    
    # Assuming the CSV has columns 'text' and 'label'
    texts = df['text'].tolist()
    labels = df['label'].tolist()  # 0 for normal, 1 for hate speech

    if 'nsina' in data_path:
        train_texts, test_texts, train_labels, test_labels = texts[0:int(len(texts)*0.9)], texts[int(len(texts)*0.9):], labels[0:int(len(labels)*0.9)], labels[int(len(labels)*0.9):]
        return train_texts, train_labels, test_texts, test_labels
        
    # Split data into train and test sets
    train_texts, test_texts, train_labels, test_labels = texts[:7501], texts[7501:], labels[:7501], labels[7501:]
    return train_texts, train_labels, test_texts, test_labels
