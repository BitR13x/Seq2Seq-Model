import torch
import json
import os
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.title_max_length = 128
        self.special_tokens = {"CLS": 3, "eos": 2, "pad": 1, "unk": 0}
        self.load_vocab()

    def load_vocab(self):
        if os.path.exists('./vocab.json'):
            with open('./vocab.json', 'r') as f:
                self.tokenizer.stoi = json.load(f)
        else:
            self.tokenizer.stoi = self.build_vocab()
            with open('./vocab.json', 'w') as f:
                json.dump(self.tokenizer.stoi, f)

        self.tokenizer.stoi.update(self.special_tokens)

    def build_vocab(self):
        vocab = set()
        for idx in range(len(self.data)):
            title = self.data.iloc[idx, 0]
            text = self.data.iloc[idx, 1]
            vocab.update(self.tokenizer(title))
            vocab.update(self.tokenizer(text))
        vocab = list(vocab)
        vocab.sort()

        return {word: i for i, word in enumerate(vocab, start=3)}

    def get_vocab_len(self):
        return len(self.tokenizer.stoi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data.iloc[idx, 0]
        text = self.data.iloc[idx, 1]

        # Tokenize the title and text
        title_tokens = self.tokenizer(title)
        text_tokens = self.tokenizer(text)

        
        # Convert tokens to tensor
        title_tensor = torch.tensor([self.tokenizer.stoi[token] for token in title_tokens]).long()
        text_tensor = torch.tensor([self.tokenizer.stoi[token] for token in text_tokens]).long()

        # Pad the sequence to max_length - 1
        title_tensor = torch.nn.functional.pad(title_tensor, (0, self.title_max_length - title_tensor.shape[0] - 1), value=self.special_tokens["pad"])
        text_tensor = torch.nn.functional.pad(text_tensor, (0, self.max_length - text_tensor.shape[0] - 1), value=self.special_tokens["pad"])
        
        # add <eos> token
        title_tensor = torch.nn.functional.pad(title_tensor, (0, 1), value=self.special_tokens["eos"])
        text_tensor = torch.nn.functional.pad(text_tensor, (0, 1), value=self.special_tokens["eos"])

        # add <CLS> token
        text_tensor = torch.cat([torch.tensor([self.special_tokens["CLS"]]), text_tensor])

        return {
            'title': title_tensor,
            'text': text_tensor
        }