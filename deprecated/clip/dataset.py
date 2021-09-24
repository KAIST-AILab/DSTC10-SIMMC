import re
import json
import subprocess

from functools import lru_cache
from typing import Optional
from pathlib import Path

import torch

from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import BatchEncoding

from clip.process import dump_metadata

GLOVE_LINK = "https://nlp.stanford.edu/data/glove.6B.zip"

class CLIPDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        with open(args.item2meta_dump_path, 'r') as f:
            data = json.load(f)

        self.dataset = list()
        for k,v in data.items():
            if k == "vocabulary":
                continue
            self.dataset.append((v['id'], v['data']))

    def __len__(self):
        return len(self.dataset)

    @lru_cache()
    def __getitem__(self, idx):
        return self.dataset[idx]


class CLIPDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.regex = re.compile(r'[\w.\[\]]+|,')

    def prepare_data(self):
        # Create data root path
        data_root = Path("./clip/data")
        if not data_root.exists():
            data_root.mkdir(parents=True, exist_ok=True)

        # Download GloVE if not exists
        glove_path = Path("./clip/data/glove.6B.zip")
        glove_text = Path("./clip/data/glove.6B.50d.txt")
        if not glove_path.exists():
            subprocess.call(["wget", GLOVE_LINK, "-P", str(data_root)])
        if not glove_text.exists():
            subprocess.call(["unzip", str(glove_path), "-d", str(data_root)])        

        dump_metadata(
            self.args.item2meta_dump_path,
            self.args.glove_path,
            self.args.metadata_path,
            self.args.glove_dim,
            self.args.separate_domain
        )
        
        with open(self.args.item2meta_dump_path, 'r') as f:
            self.item2meta = json.load(f)
    
    def setup(self, stage: Optional[str]):
        self.train_dataset = CLIPDataset(self.args)

    def collate_fn(self, batch):
        item_id, meta_data = tuple(map(list, zip(*batch)))
        tokenized = [self.regex.findall(s) for s in meta_data]
        tokenized = [[self.item2meta['vocabulary'][w] for w in s] for s in tokenized]

        max_len = max(len(s) for s in tokenized)
        for i, s in enumerate(tokenized):
            num_to_pad = max_len - len(s)
            tokenized[i] += [0] * num_to_pad
        return torch.Tensor(item_id).int(), torch.Tensor(tokenized).int()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn
        )