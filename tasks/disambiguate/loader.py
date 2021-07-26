import attr
import json

from argparse import Namespace
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

@attr.s
class DisambiguationDataset(Dataset):
    args: Namespace = attr.ib()
    load_path: str = attr.ib()
    sep_token: str = attr.ib()

    def __attrs_post_init__(self):
        self.dataset = json.load(open(self.load_path, 'r'))
        self.num_utterances = 2 * self.args.max_turns + 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[str, int]:
        """
            This is where you define the input.
        """
        dialog, lbl = self.dataset[idx]
        for turn_id, turn in enumerate(dialog):
            if self.args.use_special_tokens:
                if not turn_id % 2:
                    dialog[turn_id] = "[USER] " + turn
                else:
                    dialog[turn_id] = "[SYS] " + turn
            else:
                dialog[turn_id] = "{} ".format(self.sep_token) + turn
        txt = ' '.join(dialog[-self.num_utterances:]) 
        return txt, lbl

@attr.s
class DisambiguationCollector:
    """
        Collector class for collate_fn arg in DataLoader.
    """
    args: Namespace = attr.ib()
    tokenizer: PreTrainedTokenizerBase = attr.ib()

    def __call__(self, batch) -> Tuple[BatchEncoding, Tensor]:
        txt, lbl = map(list, zip(*batch))
        inp = self.tokenizer(
            txt,
            padding="longest",
            max_length=self.args.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return inp, Tensor(lbl).long()
        