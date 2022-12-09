import os
import sys
import re
import ast
import copy
import json
import argparse
import logging

from tqdm import tqdm

import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, BartTokenizerFast

from run_bart_multi_task_visfeature_512 import BoxEmbedding, NoCorefHead, FashionEncoderHead, FurnitureEncoderHead, VisFC, NonVisFC



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57
FASHION_SPECIAL_TOKENS = [f"<@1{i:03}>" for i in range(NUM_FASHION_ITEMS)]
FURNITURE_SPECIAL_TOKENS = [f"<@2{i:03}>" for i in range(NUM_FURNITURE_ITEMS)]

MAX_NUM_OBJ_IN_SCENE = 200
OBJECT_INDICES = [f"<{i}>" for i in range(MAX_NUM_OBJ_IN_SCENE)]

START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"
NO_COREF = "<NOCOREF>"

vision_feature = None

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def get_input_id(tokenizer, tokens):
    return tokenizer(tokens).input_ids[1:-1]

def id_converter(tokenizer):
    id2index = {get_input_id(tokenizer, index)[0]: index for index in OBJECT_INDICES}
    id2fashion_st = {get_input_id(tokenizer, st)[0]: st for st in FASHION_SPECIAL_TOKENS}
    id2furniture_st = {get_input_id(tokenizer, st)[0]: st for st in FURNITURE_SPECIAL_TOKENS}
    return id2index, id2fashion_st, id2furniture_st

def correct_action(text, correction_dict):
    for k, v in correction_dict.items():
        text = text.replace(k, v)
    return text

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def delete_elements_by_indices(list_obj, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_obj):
            list_obj.pop(idx)
    return list_obj

class RetrievalDataset(Dataset):
    def __init__(self, prompts_from_file, candidates_file, tokenizer):
        self.tokenizer = tokenizer
        lines = []
        self.original_lines = []
        self.boxes = []
        self.scene_ids = []
        
        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        exclude_indices = []
        line_ctr = 0
        # extract input sequence to BART, and bbox info to be embedded
        with open(prompts_from_file, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    # [[0.2, 0.3, 0.1, 0.2, 0.4, 0.8], [0.2, 0.1, 0.1, 0.2, 0.3, 0.1], ...]
                    if 'furniture' in line.rsplit('[', 1)[1]:
                        exclude_indices.append(line_ctr)
                    scene_ids = ast.literal_eval('[' + line.rsplit('[', 1)[1].rsplit(']', 1)[0] + ']')
                    self.scene_ids.append(scene_ids)
                    line = line.rsplit('[', 1)[0]
                    line_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", line)]
                    self.boxes.append(line_boxes)
                    line = re.sub(r"\[\([^\)]*\)\]", "", line)
                    original_line = copy.deepcopy(line)
                    original_line = re.sub(r" <SOO.*EOO>", "", original_line)
                    lines.append("<DISAM>" +line)
                    self.original_lines.append(original_line)
                    line_ctr += 1

        lines = delete_elements_by_indices(lines, exclude_indices)
        self.original_lines = delete_elements_by_indices(self.original_lines, exclude_indices)
        self.boxes = delete_elements_by_indices(self.boxes, exclude_indices)
        self.scene_ids = delete_elements_by_indices(self.scene_ids, exclude_indices)

        self.examples = tokenizer(lines, add_special_tokens=True).input_ids
        
        nocoref_id = get_input_id(tokenizer, NO_COREF)[0]
        self.nocoref = []  # [position, position, position, ...]
        self.misc = []  # [ [ {pos, is_fashion}, ... ], ...]
        id2index, id2fashion_st, id2furniture_st = id_converter(tokenizer)
        for idx, tokenized_line in enumerate(self.examples):
            tl = tokenized_line

            EOM_indices = [i for i, tokenized_id in enumerate(tl) if tokenized_id ==EOM_id]
            if EOM_indices:
                EOM_last_idx = EOM_indices[-1]
            else:
                EOM_last_idx = -1

            self.nocoref.append(tl.index(nocoref_id))

            is_fashion = True
            for token_id in tl:
                if token_id in id2fashion_st:
                    break
                if token_id in id2furniture_st:
                    is_fashion = False
                    break

            line_labels = []
            if is_fashion:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; fashion_st = id2fashion_st[tl[i+1]]
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        temp['index'] = int(item_index.replace('<', '').replace('>', ''))
                        line_labels.append(temp)
            else:
                sys.exit("FURNITURE!!!")
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i; item_index = id2index[token_id]; furniture_st = id2furniture_st[tl[i+1]]
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        line_labels.append(temp)
            self.misc.append(line_labels)
        with open(candidates_file, "rb") as f:
            self.candidates_file = json.load(f)
        
        self.candidates_file = delete_elements_by_indices(self.candidates_file, exclude_indices)
        assert len(self.candidates_file) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        asset = self.candidates_file[i]
        diag_index = asset["dialog_id"]
        turn_id = asset["turn_id"]
        candidate = self.tokenizer(asset["candidates"], padding="longest", truncation=True, return_tensors="pt")
        return torch.tensor(self.examples[i], dtype=torch.long), self.original_lines[i], self.boxes[i], self.misc[i], self.nocoref[i], \
                candidate.input_ids, candidate.attention_mask, diag_index, turn_id, self.scene_ids[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help='model dir which contains model, optimizer, scheduler weight files'
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        required=True,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--prompts_from_file",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--candidate_file",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        '--item2id',
        type=str,
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--path_output",
        type=str,
        required=True,
        help="Path to output predictions in a line separated text file.",
    )
    parser.add_argument(
        '--vision_feature',
        type=str,
        required=True,
        help='vision feature file'
    )

    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.eval_input_file = args.prompts_from_file
    set_seed(args)

    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")

    global vision_feature
    vision_feature = torch.load(args.vision_feature)

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    with open(args.add_special_tokens, "rb") as handle:
        special_tokens_dict = json.load(handle)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} tokens")
    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(args.device)

    checkpoint = torch.load(os.path.join(args.model_dir, 'aux_nets.pt'))

    box_embedding = BoxEmbedding(model.config.d_model).to(args.device)
    vis_fc = VisFC(model.config.d_model).to(args.device)
    nonvis_fc = NonVisFC(model.config.d_model).to(args.device)

    box_embedding.load_state_dict(checkpoint['box_embedding_dict'])
    vis_fc.load_state_dict(checkpoint['vis_fc'])
    nonvis_fc.load_state_dict(checkpoint['nonvis_fc'])

    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    def collate_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        original_lines = list(map(lambda x: x[1], examples))
        boxes = list(map(lambda x: x[2], examples))
        misc = list(map(lambda x: x[3], examples))
        nocoref = list(map(lambda x: x[4], examples))
        candidate = list(map(lambda x: x[5], examples)) #List[List(100, seq_len)]
        candidate_attention_mask = list(map(lambda x: x[6], examples)) #List[List(100, seq_len)]
        dialog_id = list(map(lambda x: x[7], examples))
        turn_id = list(map(lambda x: x[8], examples))
        scene_ids = list(map(lambda x: x[9], examples))
        candidate_buffer = []
        for candi in candidate:
            candidate_buffer.extend(candi)
        
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        return enc_input_pad, original_lines, boxes, misc, nocoref, candidate, candidate_attention_mask, dialog_id, turn_id, scene_ids
    
    decode_dataset = RetrievalDataset(args.prompts_from_file, args.candidate_file, tokenizer)
    decode_sampler = SequentialSampler(decode_dataset)
    decode_dataloader = DataLoader(
        decode_dataset,
        sampler=decode_sampler,
        batch_size=args.batch_size,
        collate_fn=collate_bart
    )
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)

    emb = model.model.encoder.embed_tokens

    tokenizer_id2token = {v: k for k, v in tokenizer.get_vocab().items()}
    results = []
    n_prompts = len(decode_dataset)
    for i, batch in enumerate(tqdm(decode_dataloader, desc='Retrieval...')): 
        enc_input = batch[0].to(args.device)
        original_lines = batch[1]
        boxes = batch[2] # batch, num_obj_per_line, 6
        misc = batch[3]  # batch, num_obj_per_line, dict
        nocoref = batch[4]
        candidate = batch[5][0].to(args.device) # bs(100), seq_len 
        candidate_attention_mask = batch[6][0].to(args.device) # bs(100), seq_len 
        dialog_id = batch[7]
        turn_id = batch[8]
        scene_ids = batch[9]
        
        batch_size = len(misc)

        assert batch_size == 1, "batch_size is not 1 !!"
        with torch.no_grad():
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                scenes_to_consider = scene_ids[b_idx][::-1]
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
                    item_index = misc[b_idx][obj_idx]['index']
                    visfeat = None
                    for scene_id in scenes_to_consider:
                        if scene_id in vision_feature:
                            if str(item_index) in vision_feature[scene_id]:
                                visfeat = vision_feature[scene_id][str(item_index)]
                                break
                        else:
                            print('visfeat for scene is not present!!')
                    if visfeat is None:
                        visfeat = torch.zeros(512)
                    vis_vector = torch.reshape(vis_fc(visfeat.to(args.device)), (-1,))
                    nonvis_vector = torch.reshape(nonvis_fc(emb(enc_input[b_idx][pos+1])), (-1,))
                    inputs_embeds[b_idx][pos+1] = torch.cat((vis_vector, nonvis_vector))

            encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, return_dict=True)
            context_vec = encoder_outputs[0][:, 0, :] # bs, dim
            
            response_vec = model.model.encoder(input_ids=candidate, attention_mask=candidate_attention_mask)[0][:, 0, :] # num_candi, dim
            # (num_candi, dim) @ (dim, 1) --> (num_candi, 1)
            dot_product = torch.matmul(response_vec, context_vec.T).squeeze()
            scores = torch.softmax(dot_product, dim=-1)
            
            #For teststd
            top_idx = torch.argmax(scores)
            response = tokenizer.decode(candidate[top_idx], include_special_token=False).split("<EOS>")[0][3:].strip()
            assert scores.size(0) == 100
            results.append({
                "dialog_id" : dialog_id[0],
                "candidate_scores" : [
                    {
                        "turn_id" : turn_id[0],
                        "scores" : scores.tolist(),
                        "response" : response
                    }
                ]
            })

    json.dump(results, open(args.path_output, "w"), indent=4)    
    return


if __name__ == "__main__":
    main()
