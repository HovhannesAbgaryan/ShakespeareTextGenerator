import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            self.examples = tokenizer.encode(text)

    def __len__(self):
        return len(self.examples) - self.block_size

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx : idx + self.block_size])