import os
import re
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def load_glove_embeddings(filepath, embed_dim=100):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class IMDBDataset(Dataset):
    def __init__(self, directory, embeddings, embed_dim):
        self.samples = []
        self.embeddings = embeddings
        self.embed_dim = embed_dim
        self._load_data(directory)
    
    def _load_data(self, directory):
        for label, sentiment in [('pos', 1), ('neg', 0)]:
            path = os.path.join(directory, label)
            for filename in os.listdir(path):
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    tokens = tokenize(text)
                    vectors = [torch.tensor(self.embeddings.get(token, np.zeros(self.embed_dim)), dtype=torch.float32)
                               for token in tokens]
                    self.samples.append((torch.stack(vectors), torch.tensor(sentiment)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_batch(batch):
    text_list, label_list = [], []
    for text, label in batch:
        text_list.append(text)
        label_list.append(label)
    return pad_sequence(text_list, batch_first=True), torch.tensor(label_list)

def create_data_loaders(train_dir, test_dir, binary_path, embed_dim, batch_size):
    glove_embeddings = load_glove_binary(binary_path)

    train_dataset = IMDBDataset(directory=train_dir, embeddings=glove_embeddings, embed_dim=embed_dim)
    test_dataset = IMDBDataset(directory=test_dir, embeddings=glove_embeddings, embed_dim=embed_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_loader, test_loader

def save_glove_binary(glove_path, save_path, embed_dim=100):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    np.savez_compressed(save_path, embeddings=embeddings)

def load_glove_binary(save_path):
    loaded = np.load(save_path, allow_pickle=True)
    embeddings = loaded['embeddings'].item()
    return embeddings

def preprocess_and_save_dataset(directory, glove_embeddings, embed_dim, save_path):
    dataset = []
    for label, sentiment in [('pos', 1), ('neg', 0)]:
        path = os.path.join(directory, label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = tokenize(text)
                vectors = [torch.tensor(glove_embeddings.get(token, np.zeros(embed_dim)), dtype=torch.float32)
                           for token in tokens]
                dataset.append((torch.stack(vectors), torch.tensor(sentiment)))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")

def load_preprocessed_dataset(save_path):
    with open(save_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

class IMDbPreprocessedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]