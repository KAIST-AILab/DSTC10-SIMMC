import json

from pathlib import Path
from argparse import Namespace
from functools import lru_cache
from typing import Tuple, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BatchEncoding

from baseline.process import convert_json_to_flattened, get_special_tokens

class LineTextDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        
        # Tab-separated line-by-line dataset
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.targets = data['target']

    def __len__(self):
        return len(self.targets)
    
    @lru_cache()
    def __getitem__(self, idx):
        return self.targets[idx]


class BaselineDataModule(LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_name)

    def prepare_data(self) -> None:
        # This method runs only on rank zero (main process in distributed mode)
        train_path = Path(self.args.train_processed_path)
        dev_path = Path(self.args.dev_processed_path)
        devtest_path = Path(self.args.devtest_processed_path)
        # Check for preprocessed file caches
        exists = train_path.exists() \
                    and dev_path.exists() \
                        and devtest_path.exists()
        special_tokens = get_special_tokens(
            self.args.use_belief_states,
            self.args.use_multimodal_contexts
        )
        # If there is none, preprocess then dump 
        if not exists:
            # Load necessary raw dataset
            with open(self.args.train_raw_path, 'r') as f:
                train_file = json.load(f)['dialogue_data']
            with open(self.args.dev_raw_path, 'r') as f:
                dev_file = json.load(f)['dialogue_data']
            with open(self.args.devtest_raw_path, 'r') as f:
                devtest_file = json.load(f)['dialogue_data']
            
            # Convert
            files = [train_file, dev_file, devtest_file]
            paths = [
                self.args.train_processed_path,
                self.args.dev_processed_path,
                self.args.devtest_processed_path
            ]
            out_of_vocab = set()
            for data, path in zip(files, paths):
                oov = convert_json_to_flattened(
                    data,
                    path,
                    self.args.context_length,
                    self.args.use_multimodal_contexts,
                    self.args.use_belief_states
                )
                out_of_vocab.update(oov)

            special_tokens['additional_special_tokens'].extend(
                list(out_of_vocab)
            )
            with open(self.args.tokenizer_path, 'w') as f:
                json.dump(special_tokens, f)

    def setup(self, stage: Optional[str]=None) -> None:
        with open(self.args.tokenizer_path, 'r') as f:
            special_tokens = json.load(f)
        _ = self.tokenizer.add_special_tokens(special_tokens)
        self.train_dataset = LineTextDataset(self.args.train_processed_path)
        self.dev_dataset = LineTextDataset(self.args.dev_processed_path)
        self.devtest_dataset = LineTextDataset(self.args.devtest_processed_path)

    def collate_fn(self, batch) -> Tuple[BatchEncoding, BatchEncoding]:
        inputs = self.tokenizer(
            batch,
            padding="longest",
            max_length=self.args.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = inputs['input_ids']
        return inputs, labels

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = [
            DataLoader(
                self.dev_dataset,
                batch_size=self.args.batch_size * 4,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            ),
            DataLoader(
                self.devtest_dataset,
                batch_size=self.args.batch_size * 4,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        ]
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.devtest_dataset,
            batch_size=self.args.batch_size * 4,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        return loader