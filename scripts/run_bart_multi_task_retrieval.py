import argparse
import logging
import os
import re
import copy
import json
import ast
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, BartTokenizerFast
from utils import api, util
import torch
import ipdb

class BoxEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(BoxEmbedding, self).__init__()
        self.box_linear = nn.Linear(6, hidden_dim)  
        self.box_layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, box_feat):
        transformed_box = self.box_layer_norm(self.box_linear(box_feat))
        return transformed_box

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

def correct_available_sizes(text):
    try:
        if 'availableSizes =' in text:
            available_sizes_str_list = [(m.start(0), m.end(0)) for m in re.finditer(r"availableSizes =", text)]
            if not available_sizes_str_list:  # empty available_sizes_str_list: in case of (availableSizes)
                return text
            availableSizes_idx = available_sizes_str_list[0][1]
            start_bracket_idx = -1
            end_bracket_idx = -1
            for i in range(70):
                if text[availableSizes_idx+i] == '[':
                    start_bracket_idx = availableSizes_idx+i
                if text[availableSizes_idx+i] == ']':
                    end_bracket_idx = availableSizes_idx+i
                if start_bracket_idx != -1 and end_bracket_idx != -1:
                    break
            assert start_bracket_idx != -1 and end_bracket_idx != -1, f"ERROR AT def correct_available_sizes!!\n{text}"
            list_str = text[start_bracket_idx:end_bracket_idx].replace("'", "")
            list_str = str([element for element in re.findall(r"[A-Z]+", list_str)])
            return text[:start_bracket_idx] + list_str + text[end_bracket_idx:]
        else:
            return text
    except:
        print('text:', text)

def remove_bos_eos_startequal(text):
    text = text.replace('</s>', '').replace('<s>', '').replace('=====', '')
    return text

def replace_special_chars(text):
    def rep(match_re_obj):
        return match_re_obj.group(0).replace('<','').replace('>','')
    text = re.sub("<[0-9]+>", rep, text)
    available_sizes_st_list = [('<A>', 'XS'), ('<B>', 'S'), ('<C>', 'M'), ('<D>', 'L'), ('<E>', 'XL'), ('<F>', 'XXL')]
    for size_tuple in available_sizes_st_list:
        text.replace(size_tuple[0], size_tuple[1])
    return text


def insert_coref(text, coref_chars: list):
    """ coref_chars: [<11>, <44>, ...] """
    coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
    coref_list = [int(coref.replace('<', '').replace('>', '')) for coref in coref_chars]
    coref_str = str(coref_list).replace('[', '< ').replace(']',' >') if coref_list else '<  >'
    return text[:coref_pos_start+1] + ' ' + coref_str + ' <EOB>' + text[coref_pos_end:]


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class RetrievalDataset(Dataset):
    def __init__(self, prompts_from_file, candidates_file, tokenizer):
        self.tokenizer = tokenizer
        lines = []
        self.original_lines = []
        self.boxes = []

        vocab2id = tokenizer.get_vocab()
        id2vocab = {v: k for k, v in vocab2id.items()}
        SOM_id = vocab2id[START_OF_MULTIMODAL_CONTEXTS]
        EOM_id = vocab2id[END_OF_MULTIMODAL_CONTEXTS]

        # extract input sequence to BART, and bbox info to be embedded
        with open(prompts_from_file, encoding="utf-8") as f:
            for line in f.read().splitlines():
                if (len(line) > 0 and not line.isspace()):
                    # [[0.2, 0.3, 0.1, 0.2, 0.4, 0.8], [0.2, 0.1, 0.1, 0.2, 0.3, 0.1], ...]
                    line_boxes = [ast.literal_eval(position.replace('(', '').replace(')', '')) for position in re.findall(r"\[\([^\)]+\)\]", line)]
                    self.boxes.append(line_boxes)
                    line = re.sub(r"\[\([^\)]*\)\]", "", line)
                    original_line = copy.deepcopy(line)
                    original_line = re.sub(r" <SOO.*EOO>", "", original_line)
                    lines.append("<DISAM>" +line)
                    self.original_lines.append(original_line)
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
                        
                        line_labels.append(temp)
            else:
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
        assert len(self.candidates_file) == len(self.examples)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        asset = self.candidates_file[i]
        diag_index = asset["dialog_id"]
        turn_id = asset["turn_id"]
        candidate = self.tokenizer(asset["candidates"], padding="longest", truncation=True, return_tensors="pt")
        return torch.tensor(self.examples[i], dtype=torch.long), self.original_lines[i], self.boxes[i], self.misc[i], self.nocoref[i], \
                candidate.input_ids, candidate.attention_mask, diag_index, turn_id


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

    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.eval_input_file = args.prompts_from_file
    set_seed(args)

    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")



    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    with open(args.add_special_tokens, "rb") as handle:
        special_tokens_dict = json.load(handle)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} tokens")
    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(args.device)

    checkpoint = torch.load(os.path.join(args.model_dir, 'aux_nets.pt'))

    box_embedding = BoxEmbedding(model.config.d_model).to(args.device)
    box_embedding.load_state_dict(checkpoint['box_embedding_dict'])
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
        
        candidate_buffer = []
        for candi in candidate:
            candidate_buffer.extend(candi)
        
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        return enc_input_pad, original_lines, boxes, misc, nocoref, candidate, candidate_attention_mask, dialog_id, turn_id
    
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
        batch_size = len(misc)

        assert batch_size == 1, "batch_size is not 1 !!"
        with torch.no_grad():
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
            encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, return_dict=True)
            context_vec = encoder_outputs[0][:, 0, :] # bs, dim
            
            response_vec = model.model.encoder(input_ids=candidate, attention_mask=candidate_attention_mask)[0][:, 0, :] # num_candi, dim
            # (num_candi, dim) @ (dim, 1) --> (num_candi, 1)
            dot_product = torch.matmul(response_vec, context_vec.T).squeeze()
            scores = torch.softmax(dot_product, dim=-1)
            assert scores.size(0) == 100
            results.append({
                "dialog_id" : dialog_id[0],
                "candidate_scores" : [
                    {
                        "turn_id" : turn_id[0],
                        "scores" : scores.tolist()
                    }
                ]
            })

    json.dump(results, open(args.path_output, "w"), indent=4)    
    return
if __name__ == "__main__":
    main()
