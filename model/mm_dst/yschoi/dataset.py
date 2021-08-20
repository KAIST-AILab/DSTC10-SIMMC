import os
import json
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from backbone import NestedTensor
from torch import nn

from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
from PIL import Image, ImageFile


class SceneDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, image_root_dir: str, block_size=512
    ):

        print(f"Data Directory : {file_path}")
        print(f"Image Root Directory: {image_root_dir}")
        self.image_root_dir = image_root_dir
        
        assert os.path.isfile(file_path)
        with open(file_path, "rb") as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
                
    def __len__(self):
        return len(self.data.keys())
    
    def __getitem__(self, i):
        asset = self.data[str(i)]
        image = os.path.join(self.image_root_dir, asset["image"])
        im = Image.open(image).convert("RGB")
        w, h = im.size
        im = im.resize((w // 4, h // 4))
        im_tensor = torch.from_numpy(np.array(im) / 255.).permute(2, 0, 1)

        objects = asset["accum_obj"] # list 
        objects = self.tokenizer.convert_tokens_to_ids(objects)

        num_previous_objs = len(objects) - len(asset["bbox"])
        bbox = [[-2., -2., -2., -2.]] * num_previous_objs  + asset["bbox"]
        scene_objs_mask = [0]*num_previous_objs + [1]*len(asset["bbox"])
        objs_mask = [1]*len(objects)
        predict = asset["predict"] # str
        belief = asset["belief"]
        # target = asset["target"] # str
        # label = asset["label"]
        # coref_label = asset["coref_label"]
        return {
            'image' : im_tensor.float(),
            'predict' : predict,
            'belief' : belief,
            # 'target' : target,
            # 'label' : label,
            'objects' : torch.tensor(objects, dtype=torch.long), 
            'bbox' : torch.from_numpy(np.array(bbox)), # num_objects, 4
            'scene_objs_mask' : torch.from_numpy(np.array(scene_objs_mask)).float(), # num_objs,
            "objs_mask" : torch.from_numpy(np.array(objs_mask)).float(), # num_objs
            # "coref_label" : torch.from_numpy(np.array(coref_label)).float() # num_objs
        }

import torch
import argparse
from parse import parse

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    args = parse(parser)       
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, cache_dir=None
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, cache_dir=None
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError(
                "Additional special tokens file {args.add_special_tokens} not found}"
            )
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)        
        vocabs = tokenizer.get_vocab()
        vocab_size = len(tokenizer)
        for k, v in vocabs.items():
            if v in range(10):
                print(f"{v}: {k}")

    def collate_bart(examples: Dict):

        predict = list(map(lambda x: x['predict'], examples)) # List[str]
        # target = list(map(lambda x: x['target'], examples)) # List[str]
        # label = list(map(lambda x: x['label'], examples)) # List[str]
        belief = list(map(lambda x: x['belief'], examples)) # List[str]
        imgs = NestedTensor.from_tensor_list(list(map(lambda x: x["image"], examples)))
        # objects = list(map(lambda x: x['objects'], examples))
        bbox = list(map(lambda x: x['bbox'], examples))
        # scene_objs_mask = list(map(lambda x: x['scene_objs_mask'], examples))
        # objs_mask = list(map(lambda x: x['objs_mask'], examples))
        # coref_label = list(map(lambda x: x['coref_label'], examples))
        
        assert tokenizer._pad_token is not None
        predict = tokenizer.batch_encode_plus(predict, padding="longest", return_tensors="pt") # Dict
        # label = tokenizer.batch_encode_plus(label, padding="longest", return_tensors="pt").input_ids # Tensor
        belief = tokenizer.batch_encode_plus(belief, padding="longest", return_tensors="pt") #
        # target = tokenizer.batch_encode_plus(taget, padding="longest", return_tensors="pt") # Dict
        # objects_input_pad = pad_sequence(objects, batch_first=True, padding_value=tokenizer.convert_tokens_to_ids("<@0>")) # ids 50281
        bbox_pad = pad_sequence(bbox, batch_first=True) # padding space occupied with 0
        # scene_objs_mask_pad = pad_sequence(scene_objs_mask, batch_first=True) # paddig space occupied with 0
        # objs_mask_pad = pad_sequence(objs_mask, batch_first=True) # paddig space occupied with 0
        # coref_label_pad = pad_sequence(coref_label, batch_first=True, padding_value=0) # padding space occupied with 0
        return {
            "predict" : predict,
            # "label" : label,
            "belief" : None, #belief,
            "image": imgs,
            # "objects" : objects_input_pad,
            'bbox' : bbox_pad,
            # "scene_objs_mask": scene_objs_mask_pad,
            # "objs_mask" : objs_mask_pad,
            # "coref_label" : coref_label_pad
        }

    # Data Loader
    train_dataset = SceneDataset(tokenizer, None, file_path="/ext/dstc10/yschoi/train.json", \
        image_root_dir="/ext/coco_dataset/simmc2/data")
    sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=4, collate_fn=collate_bart)
    for batch in train_dataloader:
        print(batch.keys())
        break


