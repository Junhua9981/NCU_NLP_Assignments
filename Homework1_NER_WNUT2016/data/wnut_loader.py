import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, Dataset
import yaml
import os
import torch

from model.embedding_model.word_embeddings import vocab, word_embedding

BATCH_SIZE = 64

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def CoNLL_parser(file_path):
    lines = []
    with open(file_path) as f:
        lines = f.readlines()
    words = []
    targets = []
    word = []
    target = []
    if lines.count("\t") < 1: # test data
        for line in lines:
            if len(line) > 1:
                words.append(line[:-1])
    else:
        for line in lines:
            if len(line) > 1:
                w, t = line[:-1].split("\t")
                word.append(w)
                target.append(t)
            else:
                words.append(word)
                targets.append(target)
                word = []
                target = []
    return words, targets

class CoNLLDataset(Dataset):
    def __init__(self, words, targets=None):
        self.words = words
        self.targets = targets

    def __getitem__(self, index):
        if self.targets == None:
            return torch.Tensor(self.words[index]).long()
        return torch.Tensor(self.words[index]).long(), torch.Tensor(self.targets[index]).long()

    def __len__(self):
        return len(self.words)
    
def PadCollate(batch):
    if type(batch[0]) is tuple: #train, evaluate
      x = [t for t, _ in batch]
      y = [t for _, t in batch]
      x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
      y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
      return x, y
    else: #test
      x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
      return x

class WNUTDataModule(pl.LightningDataModule):
    def prepare_data(self):
        pass
        
    def setup(self, stage:str):
        dataset_path = "data/datasets/" 
        idx2word = vocab
        word2idx = { u: i for i, u in enumerate(vocab)}

        idx2tag = ['<PAD>', 'O', 'B-movie', 'B-other', 'B-geo-loc', 'B-product', 'B-musicartist', 'B-company', 'I-other', 'I-tvshow', 'I-person', 'B-sportsteam', 'B-tvshow', 'I-product', 'I-facility', 'I-sportsteam', 'I-geo-loc', 'I-company', 'I-musicartist', 'B-person', 'I-movie', 'B-facility']
        tag2idx = { u: i for i, u in enumerate(idx2tag)}

        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            
            self.train_words, self.train_targets = CoNLL_parser(f"{dataset_path}/train.txt") if os.path.exists(f"{dataset_path}/train.txt") else None
            self.train_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in self.train_words]
            self.train_targets_as_int = [[tag2idx[tag] for tag in sentence] for sentence in self.train_targets]
        
        # if stage == "validate":
            self.val_words, self.val_targets = CoNLL_parser(f"{dataset_path}/dev.txt") if os.path.exists(f"{dataset_path}/dev.txt") else None
            self.val_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in self.val_words]
            self.val_targets_as_int = [[tag2idx[tag] for tag in sentence] for sentence in self.val_targets]
        
        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.test_words, self.test_targets = CoNLL_parser(f"{dataset_path}/test-submit.txt") if os.path.exists(f"{dataset_path}/test-submit.txt") else None
            self.test_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in self.test_words]

    def train_dataloader(self):
        
        train_dataset = CoNLLDataset(self.train_words_as_int, self.train_targets_as_int)
        train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, collate_fn= PadCollate, shuffle= True)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = CoNLLDataset(self.val_words_as_int, self.val_targets_as_int)
        val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, collate_fn= PadCollate, shuffle= False)
        return val_loader
    
    def test_dataloader(self):
        test_dataset = CoNLLDataset(self.test_words_as_int)
        test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, collate_fn= PadCollate, shuffle= False)
        return test_loader