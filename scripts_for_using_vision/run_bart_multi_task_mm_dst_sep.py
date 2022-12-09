import os
import re
import ast
import copy
import json
import argparse
import logging

from tqdm import tqdm

import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, BartTokenizerFast

from run_bart_multi_task import BoxEmbedding, NoCorefHead, FashionEncoderHead, FurnitureEncoderHead

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
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
    SIZES =['<A>', '<B>', '<C>', '<D>', '<E>', '<F>']
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
            new_list = []
            for size in SIZES:
                if size in list_str:
                    new_list.append(size)
            new = ", ".join(new_list)
            return text[:start_bracket_idx] + '['+new + text[end_bracket_idx:]
        else:
            return text
    except:
        print('text:', text)

def remove_bos_eos_startequal(text):
    text = text.split("</s>")[0].replace('<s>', '')
    return text

def replace_special_chars(text):
    def rep(match_re_obj):
        return match_re_obj.group(0).replace('<','').replace('>','')
    available_sizes_st_list = [('<A>', "'XS'"), ('<B>', "'S'"), ('<C>', "'M'"), ('<D>', "'L'"), ('<E>', "'XL'"), ('<F>', "'XXL'")]
    for size_tuple in available_sizes_st_list:
        text = text.replace(size_tuple[0], size_tuple[1])
    text = re.sub("<[0-9]+>", rep, text)
    return text

def insert_coref(text, coref_chars: list):
    """ coref_chars: [<11>, <44>, ...] """
    try:
        coref_pos_start, coref_pos_end = [(m.start(0), m.end(0)) for m in re.finditer(r"\) *<EOB>", text)][0]
    
    except:
        ipdb.set_trace()
        
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

class GenerationDataset(Dataset):
    def __init__(self, prompts_from_file, tokenizer):

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
                    lines.append(line)
                    self.original_lines.append(original_line)
        encode_text = tokenizer(lines, add_special_tokens=True)
        self.examples = encode_text.input_ids
        self.examples_attention_mask = encode_text.attention_mask
        
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
                        pos = i
                        temp['is_fashion'] = True
                        temp['pos'] = pos
                        
                        line_labels.append(temp)
            else:
                for i, token_id in enumerate(tl):
                    if token_id in id2index and i > EOM_last_idx:  # this token is for item index
                        temp = dict()
                        pos = i
                        temp['is_fashion'] = False
                        temp['pos'] = pos
                        line_labels.append(temp)
            self.misc.append(line_labels)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.examples_attention_mask[i], dtype=torch.long), self.original_lines[i], self.boxes[i], self.misc[i], self.nocoref[i]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help='model dir which contains model, optimizer, scheduler weight files'
    )
    parser.add_argument(
        "--prompts_from_file",
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
        default=36
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        required=True,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--correct_act",
        type=str,
        default=None,
        help="correct wrongly generated action with correct_act dictionary",
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
    model.config.decoder_start_token_id = 0
    model.to(args.device)

    checkpoint = torch.load(os.path.join(args.model_dir, 'aux_nets.pt'))
    box_embedding = BoxEmbedding(model.config.d_model).to(args.device)
    nocoref_head = NoCorefHead(model.config.d_model).to(args.device)
    fashion_enc_head = FashionEncoderHead(model.config.d_model).to(args.device)
    furniture_enc_head = FurnitureEncoderHead(model.config.d_model).to(args.device)
    
    box_embedding.load_state_dict(checkpoint['box_embedding_dict'])
    nocoref_head.load_state_dict(checkpoint['nocoref_head_dict'])
    fashion_enc_head.load_state_dict(checkpoint['fashion_enc_head'])
    furniture_enc_head.load_state_dict(checkpoint['furniture_enc_head'])
    
    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    def collate_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        original_lines = list(map(lambda x: x[2], examples))
        boxes = list(map(lambda x: x[3], examples))
        misc = list(map(lambda x: x[4], examples))
        nocoref = list(map(lambda x: x[5], examples))
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)
        return enc_input_pad, enc_attention_pad, original_lines, boxes, misc, nocoref
    
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)
    
    decode_dataset = GenerationDataset(args.prompts_from_file, tokenizer)
    decode_sampler = SequentialSampler(decode_dataset)
    decode_dataloader = DataLoader(
        decode_dataset,
        sampler=decode_sampler,
        batch_size=args.batch_size,
        collate_fn=collate_bart
    )

    tokenizer_id2token = {v: k for k, v in tokenizer.get_vocab().items()}
    results = []
    results_coref_replaced = []
    n_prompts = len(decode_dataset)
    for i, batch in enumerate(tqdm(decode_dataloader, desc='Decoding')):  # should be 1-batchsized batch
        
        enc_input = batch[0].to(args.device)
        enc_input_attention = batch[1].to(args.device)
        original_lines = batch[2]
        boxes = batch[3] # batch, num_obj_per_line, 6
        misc = batch[4]  # batch, num_obj_per_line, dict
        nocoref = batch[5]
        batch_size = len(misc)
        # assert batch_size == 1, "batch_size is not 1 !!"
        with torch.no_grad():
            inputs_embeds = model.model.encoder.embed_tokens(enc_input) * model.model.encoder.embed_scale
            for b_idx in range(batch_size):  # in a batch
                box_embedded = box_embedding(torch.tensor(boxes[b_idx]).to(args.device))  # (num_obj_per_line, d_model)
                for obj_idx in range(len(misc[b_idx])):
                    pos = misc[b_idx][obj_idx]['pos']
                    inputs_embeds[b_idx][pos] += box_embedded[obj_idx]
            encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, attention_mask=enc_input_attention, return_dict=True)  # check this line
        
        nocoref_logits = torch.stack([nocoref_head(encoder_outputs.last_hidden_state[b_idx][nocoref[b_idx]]) for b_idx in range(batch_size)])
        is_nocoref = nocoref_logits.argmax(dim=1).bool()

        coref_obj_list = []
        coref_check = []
        for b_idx in range(batch_size):
            coref_obj_each_batch = []
            for obj_idx in range(len(misc[b_idx])):
                pos = misc[b_idx][obj_idx]['pos']
                # hidden_concat: (num_obj, 2*model)
                if obj_idx == 0:
                    hidden_concat = torch.reshape(encoder_outputs.last_hidden_state[b_idx][pos:pos+2], (1,-1))
                else:
                    hidden_concat = torch.cat([hidden_concat, torch.reshape(encoder_outputs.last_hidden_state[b_idx][pos:pos+2], (1,-1))], dim=0)
            
            objs_pos = [misc[b_idx][obj_idx]['pos'] for obj_idx in range(len(misc[b_idx]))]
            obj_indices = [tokenizer_id2token[enc_input[b_idx][pos].item()] for pos in objs_pos]  # ex) [<11>, <41>, ...]

            is_fashion = misc[b_idx][0]['is_fashion']
            if is_fashion:
                coref, size, available_sizes, brand, color, pattern, sleeve_length, \
                asset_type, type_, price, customer_review = fashion_enc_head(hidden_concat)
            else:
                coref, brand, color, materials, type_, price, customer_review = furniture_enc_head(hidden_concat)

            coref_predict = coref.argmax(dim=1).tolist()  # (num_objs)
            for i, coref_signal in enumerate(coref_predict):
                if coref_signal:
                    coref_obj_each_batch.append(obj_indices[i])
            coref_obj_list.append(coref_obj_each_batch)
            coref_check.append(True if len(coref_obj_each_batch) > 0 else False)
        
        check = torch.logical_and(is_nocoref.cpu(), torch.tensor(coref_check, dtype=torch.bool))
        if check.any():
            print("is_nocoref and object is both on!!! THIS SHOULD NOT HAPPEN!")
            idx = (check == True).nonzero(as_tuple=True)[0].tolist()
            for i in idx:
                coref_obj_list[i] = []


        output_sequences = model.generate(
            max_length=args.length + inputs_embeds.size()[1],
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            encoder_outputs=encoder_outputs)

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
        
        generated_sequences = []
        generated_sequences_coref_replaced = []
        
        predicts = tokenizer.batch_decode(output_sequences, include_special_token=True)
        for sequence_idx, text in enumerate(predicts):
            if sequence_idx == 0:
                print(
                    "=== GENERATED SEQUENCE {sequence_idx}, {promt_idx}/{n_prompts} ===".format(
                        sequence_idx=sequence_idx + 1,
                        promt_idx=i + 1,
                        n_prompts=n_prompts,
                    )
                )
                print("Before replace : " + text.split("</s>")[0].strip())
            
            text = remove_bos_eos_startequal(text)
            text = correct_available_sizes(text)
            text_coref_replaced = copy.deepcopy(text)
            total_sequence = replace_special_chars(original_lines[sequence_idx] + text_coref_replaced)
            total_sequence_coref_replaced = insert_coref(total_sequence, coref_obj_list[sequence_idx])
            generated_sequences_coref_replaced.append(total_sequence_coref_replaced)

            if sequence_idx == 0:
                print('total_sequence_coref_replaced:', total_sequence_coref_replaced, '\n')

        results_coref_replaced.extend(generated_sequences_coref_replaced)
    with open(args.path_output, "w") as f_out:
        f_out.write("\n".join(results_coref_replaced))
    
    return 

if __name__ == "__main__":
    main()
