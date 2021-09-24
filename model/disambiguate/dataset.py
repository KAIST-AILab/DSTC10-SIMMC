import json

from typing import Optional
from functools import lru_cache

import torch

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from process import get_special_tokens, convert_json_to_flattened

class LineTextDataset(Dataset):
    def __init__(self, dialog_path: str):
        super().__init__()
        
        # Dialogue -> Dict[str, Dict[str, str]]
        with open(dialog_path, 'r') as f:
            self.dialog = json.load(f)

    def __len__(self) -> int:
        return len(self.dialog)
    
    @lru_cache()
    def __getitem__(self, idx):
        sample = self.dialog[idx]
        return (sample['dialog_id'], sample['turn_id'], sample['source'], sample['disambiguation_label'])


class DisambiguationDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_checkpoint)

    def prepare_data(self) -> None:
        special_tokens = get_special_tokens(self.args.generate_sys_attr)
        # Load necessary raw dataset
        with open(self.args.train_raw_path, 'r') as f:
            train_file = json.load(f)['dialogue_data']
        with open(self.args.dev_raw_path, 'r') as f:
            dev_file = json.load(f)['dialogue_data']
        with open(self.args.devtest_raw_path, 'r') as f:
            devtest_file = json.load(f)['dialogue_data']
        # Convert
        files = [train_file, dev_file, devtest_file]
        paths = \
            [
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
                self.args.generate_sys_attr
            )
            out_of_vocab.update(oov)
        # Dump special tokens
        special_tokens['additional_special_tokens'].extend(
            list(out_of_vocab)
        )
        with open(self.args.tokenizer_path, 'w') as f:
            json.dump(special_tokens, f, indent=4)
     
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
        (
            dialog_id,
            turn_id,
            encoder_inputs,
            disambiguation_labels
        ) = tuple(map(list, zip(*batch)))
        
        encoder_inputs = self.tokenizer(
            encoder_inputs,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        disambiguation_labels = torch.tensor(
            disambiguation_labels, dtype=torch.long
        )
        return dialog_id, turn_id, encoder_inputs, disambiguation_labels

    def train_dataloader(self) -> DataLoader:
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
        # Padding done on left for generation
        loader = DataLoader(
            self.devtest_dataset,
            batch_size=self.args.batch_size * 8,
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers
        )
        return loader