import json

from pathlib import Path
from argparse import Namespace
from functools import lru_cache
from typing import Tuple, Optional

import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from process import convert_json_to_flattened, get_special_tokens


class LineTextDataset(Dataset):
    '''
        A line-by-line dataset that takes each line as individual sample (that is, we do not split lines into blocks of fixed size).

        Args:
            data_path <str>: path to dataset (processed)
            is_train <bool>: flag for training
                - for training, returns (target, target)
                - else, returns (source, target) for evaluation
    '''
    def __init__(self, data_path: str, is_train: bool=True):
        super().__init__()
        
        # Flag for training
        self.is_train = is_train
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.sources = data['source']
        self.targets = data['target']
        self.disambiguation_labels = data['disambiguation_label']

    def __len__(self) -> int:
        return len(self.sources)
    
    @lru_cache()
    def __getitem__(self, idx):
        return (
            self.sources[idx], self.targets[idx],
            self.disambiguation_labels[idx]
        )


class MultiTaskDataModule(LightningDataModule):
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
        # If there is none, preprocess then dump 
        if not exists:
            special_tokens = get_special_tokens(
                self.args.use_belief_states,
                self.args.use_multimodal_contexts
            )
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
                    self.args.use_belief_states,
                    self.args.generate_sys_attr
                )
                out_of_vocab.update(oov)
            # Dump special tokens
            special_tokens['additional_special_tokens'].extend(
                list(out_of_vocab)
            )
            with open(self.args.tokenizer_path, 'w') as f:
                json.dump(special_tokens, f)
        else:
            # Else, load the special tokens
            with open(self.args.tokenizer_path, 'r') as f:
                special_tokens = json.load(f)
        # Add special tokens
        _ = self.tokenizer.add_special_tokens(special_tokens)

    def setup(self, stage: Optional[str]=None) -> None:
        if stage in ('fit'):
            self.train_dataset = LineTextDataset(
                self.args.train_processed_path
            )
            self.dev_dataset = LineTextDataset(
                self.args.dev_processed_path
            )
            self.devtest_dataset = LineTextDataset(
                self.args.devtest_processed_path
            )
        else:
            self.devtest_dataset = LineTextDataset(
                self.args.devtest_processed_path, False
            )

    def collate_fn(self, batch):
        encoder_inputs, decoder_inputs, disambiguation_labels = tuple(map(list, zip(*batch)))
        encoder_inputs = self.tokenizer(
            encoder_inputs,
            padding="longest",
            max_length=self.args.max_length,
            truncation=True,
            return_tensors="pt"
        )
        decoder_inputs = self.tokenizer(
            decoder_inputs,
            padding="longest",
            max_length=self.args.max_length,
            truncation=True,
            return_tensors="pt"
        )
        disambiguation_labels = torch.Tensor(disambiguation_labels).long()
        return encoder_inputs, decoder_inputs, disambiguation_labels

    def train_dataloader(self) -> DataLoader:
        self.tokenizer.padding_size = 'right'
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = [
            DataLoader(
                self.dev_dataset,
                batch_size=self.args.batch_size * 2,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers
            ),
            DataLoader(
                self.devtest_dataset,
                batch_size=self.args.batch_size * 2,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers
            )
        ]
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.devtest_dataset,
            batch_size=self.args.batch_size * 12,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers
        )
        return loader