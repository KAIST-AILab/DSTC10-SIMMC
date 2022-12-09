import os
import re
import ast
import copy
import json
import argparse
import logging
import ipdb
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
def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
def plot_heat_map(mma, target_labels, source_labels, name):
    fig, axs = plt.subplots(1,5, figsize=(23, 6), gridspec_kw={"width_ratios":[1, 1, 1, 1, 0.1]})
    # fig, axs = plt.subplots(2,2, figsize=(9, 8))
    for i in range(4):
        heatmap = axs[i].pcolor(mma[i], cmap=plt.cm.Blues)
        axs[i].set_xticks(numpy.arange(mma[i].shape[1]) + 0.5, minor=False) 
        axs[i].set_yticks(numpy.arange(mma[i].shape[0]) + 0.5, minor=False) 
        axs[i].set_xlim(0, int(mma[i].shape[1]))
        axs[i].set_ylim(0, int(mma[i].shape[0]))
        axs[i].invert_yaxis()
        axs[i].xaxis.tick_top()
        axs[i].set_xticklabels(source_labels[i], fontsize=16, minor=False)
        axs[i].set_yticklabels(target_labels[i], fontsize=16, minor=False)
        # axs[i].xaxis.set_tick_params(rotation=50)
    
    ip = InsetPosition(axs[-2], [1.05,0,0.05,1])
    axs[-1].set_axes_locator(ip)
    cbar = fig.colorbar(heatmap, cax=axs[-1], ax=axs.ravel().tolist()[:-1])
    plt.subplots_adjust(left=0.06, bottom=0.01,  right=0.98, top=0.93, wspace=0.25, hspace=0.16)
    plt.savefig(f"{name}.png")
    plt.close()
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
        default=1
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
    
    print('checkpoint', checkpoint)

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
    
    alpha_list = []
    target_label_1_list = []
    target_label_2_list = []
    for batch_index, batch in enumerate(tqdm(decode_dataloader, desc='Decoding')):  # should be 1-batchsized batch
        
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
            encoder_outputs = model.model.encoder(inputs_embeds=inputs_embeds, attention_mask=enc_input_attention, output_attentions=True, return_dict=True)  # check this line
        
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
        
        # num obj
        soo_pos = misc[0][0]['pos'] - 3
        obj_start = misc[0][0]['pos']-1
        num_obj = enc_input[0].size(0) - misc[0][0]['pos']
        # if num_obj > 30:
        #     continue
        # if len(coref_obj_list[0]) < 2:
        #     continue
        # if batch_index < 33 : 
        #     continue
        label1 = ["169", "152", "256", "168", "258", "283", "277"]
        label2 = ["115", "167", "005", "069", "265", "188"]
        start_positions = [59, 69, 59, 73]
        if batch_index in [115, 230, 370, 613]:
        # if batch_index in [230, 613]:
            print(coref_obj_list[0])
            start_pos = start_positions[len(alpha_list)]
            last_pos = soo_pos -1            
            # ipdb.set_trace()
            # find out start pos
            for b in range(args.batch_size):
                target_label = tokenizer.convert_ids_to_tokens(enc_input[b])
                target_label = [i.replace('Ġ', '') for i in target_label]
                alpha = encoder_outputs.attentions[-1][b][3].cpu().data.numpy() # seq_len, seq_len --> work 
                alpha_list.append(copy.deepcopy(alpha[start_pos:last_pos, obj_start+2:-10:3]))
                target_label_1_list.append(copy.deepcopy(target_label[start_pos:last_pos]))
                if batch_index in [115, 230]:
                    # target_label_2_list.append(copy.deepcopy(target_label[obj_start+2:-10:3]))
                    target_label_2_list.append(copy.deepcopy(label1))
                else:
                    target_label_2_list.append(copy.deepcopy(label2))
            # ipdb.set_trace()
            # plot_heat_map(alpha_list[-1], target_label_1_list[-1], target_label_2_list[-1], f"last-layer-5th-head-fashion{len(alpha_list)}th")
        # if batch_index == 230:
            # plot_heat_map(alpha_list[:2], target_label_1_list[:2], target_label_2_list[:2], f"last-layer-5th-head-1st")
        if len(alpha_list) == 4:
            # print('alpha_list', alpha_list)
            plot_heat_map(alpha_list, target_label_1_list, target_label_2_list, f"last-layer-3th-head-final-beta_original")
            break
    
if __name__ == "__main__":
    main()

    # python attention_map.py   \
    # --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt   \
    # --path_output=../devtest_results/dstc10-simmc-devtest-pred-subtask-3.txt   \
    # --item2id=item2id.json   \
    # --add_special_tokens=../data_object_special/simmc_special_tokens.json   \
    # --model_dir=../multi_task/checkpoint-22000
