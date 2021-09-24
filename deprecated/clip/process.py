'''
    Quick metadata extraction script for CLIP experiments. 
'''
import re
import json

from typing import List, Dict
from pathlib import Path
from os.path import join

import torch
import numpy as np

# Non-visual
BRAND = "brand"
REVIEW = "review"
PRICE = "price"
SIZE = "size"
MATERIALS = "materials" 
AVAILABLE = "available"

# Visual
ASSET = "asset"
COLOR = "color"
PATTERN = "pattern"
SLEEVE = "sleeve"

META_FASHION_FORMAT = "{brand_token} {brand} {review_token} {review} {price_token} {price} {size_token} {size} {available_token} {available} {asset_token} {asset} {color_token} {color} {pattern_token} {pattern} {sleeve_token} {sleeve}"

META_FURNITURE_FORMAT = "{brand_token} {brand} {review_token} {review} {price_token} {price} {asset_token} {asset} {materials_token} {materials}"

DOMAIN = {
    'fashion': 1,
    'furniture': 2
}

CAMEL2WORD = re.compile(r'(?<!^)(?=[A-Z])')


def load_glove_model(file_path: str, vocabs: List) -> Dict:
    glove_model = {}
    with open(file_path,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            if word not in vocabs:
                continue
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = torch.tensor(embedding)
        
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def dump_metadata(
    dump_path: str,
    glove_path: str,
    metadata_path: str,
    glove_dim: int,
    separate_domain: bool=False
) -> None:
    regex = re.compile(r'[\w.\[\]]+|,')
    vocabs = set()
    metadata_files = [
        Path(
            join(metadata_path, "{}_prefab_metadata_all.json".format(k))
        ) for k in DOMAIN.keys()
    ]
    # Mapping from item to token id
    item2vector = dict()
    cnt = 1
    
    for metadata_file in metadata_files:
        # Load metadata
        metadata = json.load(open(metadata_file, "r"))
        domain = DOMAIN['fashion'] if metadata_file.stem.startswith("fashion") else DOMAIN['furniture']

        for prefab, item_dict in metadata.items():
            # item_vector = "{}{:03d}".format(domain, cnt)
            # special_tokens_dict['additional_special_tokens'].append(special_token)
            item_vector = int(cnt)
            stringified = ""
            if domain == DOMAIN['fashion']:
                if separate_domain:
                    stringified += "fashion_domain "
                stringified += META_FASHION_FORMAT.format(
                    brand_token=BRAND,
                    brand=item_dict[BRAND],
                    review_token=REVIEW,
                    review=item_dict['customerReview'],
                    price_token=PRICE,
                    price=item_dict[PRICE],
                    size_token=SIZE,
                    size=item_dict[SIZE],
                    available_token=AVAILABLE,
                    available=str(item_dict['availableSizes']).replace('\'',''),
                    asset_token=ASSET,
                    asset=item_dict['type'],
                    color_token=COLOR,
                    color=item_dict[COLOR],
                    pattern_token=PATTERN,
                    pattern=item_dict[PATTERN],
                    sleeve_token=SLEEVE,
                    sleeve=item_dict['sleeveLength']
                ).strip().lower()
            else:
                if separate_domain:
                    stringified += "furniture_domain "
                stringified += META_FURNITURE_FORMAT.format(
                    brand_token=BRAND,
                    brand=item_dict[BRAND],
                    review_token=REVIEW,
                    review=item_dict['customerRating'],
                    price_token=PRICE,
                    price=item_dict[PRICE].strip('$'),
                    asset_token=ASSET,
                    asset=item_dict['type'],
                    materials_token=MATERIALS,
                    materials=item_dict[MATERIALS]
                ).strip().lower()
            item2vector[prefab] = {
                'id': item_vector,
                'data': stringified
            }
            vocabs = vocabs.union(set(regex.findall(stringified)))
            cnt += 1
    
    # Sort vocabular and create a mapping
    vocabs = sorted(list(vocabs))
    word2id = {v: k for k,v in enumerate(vocabs, 1)}
    
    # Dump item to metadata
    print("Vocabulary size: ", len(vocabs))
    item2vector['vocabulary'] = word2id
    with open(dump_path, 'w') as f:
        json.dump(item2vector, f)

    # Extract glove
    glove_dict = load_glove_model("./clip/data/glove.6B.{}d.txt".format(glove_dim), word2id)
    for k, v in word2id.items():
        # Randomly initialize vocab not in GloVE
        if k not in glove_dict:
            glove_dict[k] = torch.randn(glove_dim)
    
    # Dump embedding
    embedding = list(torch.zeros([1, glove_dim]))
    for k, v in sorted(glove_dict.items()):
        embedding.append(v)
    embedding = torch.stack(embedding, 0)
    torch.save(embedding, glove_path)