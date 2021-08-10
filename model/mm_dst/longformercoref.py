import sys
import random
import glob
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import argparse
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from utils.api import PromptAPI
from transformers import (
    LongformerTokenizer,
    LongformerForTokenClassification,
    AdamW, 
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

OBJ_TOKEN = '<OBJ>'
NO_COREF_TOKEN = '<NOCOREF>'



logger = logging.getLogger(__name__)

def set_seed(seed ,n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class CorefLongformerDataset(Dataset):
    def __init__(self, tokenizer, split):
        try:
            obj_token_id = tokenizer.get_vocab()[OBJ_TOKEN]
            no_coref_token_id = tokenizer.get_vocab()[NO_COREF_TOKEN]
        except KeyError:
            print(f"{OBJ_TOKEN} and {NO_COREF_TOKEN} should have been added to tokenizer!!")
            sys.exit(1)

        self.prompt_api = PromptAPI(dial_split=split)
        dial_data = self.prompt_api.dial_data_returner(len_history=2)
        coref_data = []
        coref_objs = []
        for one_dial in dial_data:
            # fetching object info
            attrs_to_consider = ['asset_type', 'color', 'pattern', 'sleeve_length', 'size', 
                'available_sizes', 'brand', 'customer_review']
            scene_str_dict = dict()
            for scene_idx, scene_obj_dict in one_dial['scene_objects'].items():
                scene_obj_str = ''
                obj_indices = sorted(list(scene_obj_dict.keys()))
                for obj_idx in obj_indices:
                    obj_attrs = scene_obj_dict[obj_idx]
                    obj_attrs_str = ' '.join([str(obj_attrs[attr]) if str(obj_attrs[attr]) else 'None' for attr in attrs_to_consider])
                    obj_attrs_str = f' {OBJ_TOKEN} ' + str(obj_idx) + ' : ' + obj_attrs_str
                    scene_obj_str += obj_attrs_str
                scene_str_dict[scene_idx] = scene_obj_str

            for turn_idx, one_turn in enumerate(one_dial['dialogues']):
                one_turn_str = ' '.join(one_turn['context_with_obj'])
                coref_objs.append(one_turn['belief']['objects'])
                this_turn_scene_idx = 0
                for scene_idx in one_dial['scene_objects'].keys():
                    if turn_idx >= scene_idx:
                        this_turn_scene_idx = scene_idx
                    else: 
                        break
                one_turn_str += ' ' + NO_COREF_TOKEN + scene_str_dict[this_turn_scene_idx] 
                coref_data.append(one_turn_str)

        # 짤리는 object에 대해선 coref못잡는 일이 발생할수 있지만, but we use longformer here
        self.enc_input = tokenizer(coref_data, add_special_tokens=True).input_ids

        # set token_classification_labels
        self.token_classification_labels = []
        # set global attention
        self.global_attention = []

        for data_idx, coref_obj in enumerate(coref_objs):  # coref_objs is a list
            coref_obj_label_list = torch.zeros(len(self.enc_input[data_idx]))
            this_line_global_attention = torch.zeros(len(self.enc_input[data_idx]))

            # for global attention, set index 1
            for input_id_idx, input_id in enumerate(self.enc_input[data_idx]): 
                if input_id == no_coref_token_id: 
                    this_line_global_attention[:input_id_idx] = 1
                    break
                    
            # for token_classification_labels, set <OBJ> token output 1
            if not coref_obj:
                for input_id_idx, input_id in enumerate(self.enc_input[data_idx]): 
                    if input_id == no_coref_token_id:
                        coref_obj_label_list[input_id_idx] = 1
                        break
                continue
            else:
                obj_token_id_counter = 0
                for input_id_idx, input_id in enumerate(self.enc_input[data_idx]):
                    if input_id == obj_token_id:
                        if obj_token_id_counter in coref_obj:
                            coref_obj_label_list[input_id_idx] = 1
                        obj_token_id_counter += 1

            self.token_classification_labels.append(coref_obj_label_list)            
            self.global_attention.append(this_line_global_attention)

    def __len__(self):
        return len(self.enc_input)
    
    def __getitem__(self, i):
        return  torch.tensor(self.enc_input[i] ,dtype=torch.long), \
                torch.tensor(self.token_classification_labels[i], dtype=torch.long), \
                torch.tensor(self.global_attention[i], dtype=torch.float)


def handle_incomplete_batch(tokenizer, split):
    dataset = CorefLongformerDataset(tokenizer, split)
    n = len(dataset) % BATCH_SIZE
    if n != 0:
        dataset.enc_input = dataset.enc_input[:-n]
        dataset.token_classification_labels = dataset.token_classification_labels[:-n]
        dataset.global_attention = dataset.global_attention[:-n]
    return dataset


def evaluate(model, tokenizer, args, prefix=''):
    eval_dataset = handle_incomplete_batch(tokenizer, 'dev')
    
    def collate_longformer(examples: List[torch.Tensor]):
        enc_input = list(map(lambda x: x[0], examples))
        labels = list(map(lambda x: x[1], examples))
        global_attention_mask = list(map(lambda x: x[2], examples))
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
            labels_pad = pad_sequence(labels, batch_first=True)
            global_attention_mask_pad = pad_sequence(global_attention_mask, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)
            labels_pad = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
            global_attention_mask_pad = pad_sequence(global_attention_mask, batch_first=True, padding_value=.0)
        return enc_input_pad, labels_pad, global_attention_mask_pad
    
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=BATCH_SIZE, collate_fn=collate_longformer)
    
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    num_correct = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        enc_input = batch[0].to(args.device)
        token_classification_label = batch[1].to(args.device)
        global_attention_mask = batch[2].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=enc_input, global_attention_mask = global_attention_mask, labels=token_classification_label)
        loss = outputs.loss
        logits = outputs.logits
        values, indices = torch.max(logits, dim=-1)
        prediction = indices

        if prediction.tolist() == token_classification_label.tolist():
            num_correct += 1

        eval_loss += loss.mean().item()
        nb_eval_steps += 1

    accuracy = float(num_correct) / nb_eval_steps
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {"perplexity": perplexity, 'accuracy': accuracy}
    output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    print('eval:', result)

    return result
    

def train(model, tokenizer, args):
    train_dataset = handle_incomplete_batch(tokenizer, 'train')
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size, collate_fn=collate_longformer)
    tb_writer = SummaryWriter()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    global_step = 0
    t_total = len(train_dataloader) * args.train_epochs
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    # Check if saved optimizer or scheduler states exist
    if ( args.save_dir  
         and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
         and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.train_epochs)
    logger.info("  batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    epochs_trained = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.train()
    model.zero_grad()

    train_iterator = trange(
        epochs_trained,
        int(args.train_epochs),
        desc="Epoch",
    )

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            enc_input = batch[0].to(args.device)
            token_classification_label = batch[1].to(args.device)
            global_attention_mask = batch[2].to(args.device)
            
            outputs = model(input_ids=enc_input, global_attention_mask = global_attention_mask, labels=token_classification_label)
            loss = outputs[0]

            if step % 500 == 0:
                print('train loss', loss)

            loss.backward()

            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_step  > 0 and global_step % args.logging_steps == 0:
                results = evaluate(model, tokenizer, args)
                for key, value in results.items():
                    tb_writer.add_scalar(
                        "eval_{}".format(key), value, global_step
                    )
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
            
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = "checkpoint"
                save_dir = os.path.join(
                        args.save_dir, "{}-{}".format(checkpoint_prefix, global_step)
                    )
                os.makedirs(save_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                torch.save(args, os.path.join(save_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", save_dir)

                torch.save(
                        optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt")
                    )
                torch.save(
                    scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt")
                )
                logger.info(
                        "Saving optimizer and scheduler states to %s", save_dir
                    )
    return global_step, tr_loss / global_step



def main(args):

    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    args.device = device
    
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
    
    spcial_tokens = {"eos_token": "<EOS>", "additional_special_tokens": 
    ["USER", "SYS", OBJ_TOKEN, NO_COREF_TOKEN, "<EOB>", "<SOM>", "<EOM>", "INFORM:REFINE", "REQUEST:COMPARE", "brand", "INFORM:GET", "customerReview", "materials", 
    "customerRating", "ASK:GET", "pattern", "INFORM:DISAMBIGUATE", "availableSizes", "REQUEST:GET", "color", "sleeveLength", "size", 
    "REQUEST:ADD_TO_CART", "price", "type", "REQUEST:COMPARE", "color", "availableSizes", "REQUEST:GET", "customerReview", "type", "brand", "price", "REQUEST:ADD_TO_CART", 
    "INFORM:DISAMBIGUATE", "pattern", "INFORM:REFINE", "sleeveLength", "INFORM:GET", "size", "ASK:GET", "REQUEST:ADD_TO_CART", "brand", "customerReview", "size", "INFORM:DISAMBIGUATE", 
    "INFORM:REFINE", "ASK:GET", "REQUEST:GET", "INFORM:GET", "color", "customerRating", "price", "type", "pattern", "materials", "availableSizes", "sleeveLength", "REQUEST:COMPARE"]}
    tokenizer.add_special_tokens(spcial_tokens)

    model = LongformerForTokenClassification.from_pretrained('allenai/longformer-large-4096', num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.vocab_size = len(tokenizer)

    model.to(device)

    set_seed(args.seed)

    if args.do_train:
        global_step, tr_loss = train(model, tokenizer, args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("Saving model checkpoint to %s", args.save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        torch.save(args, os.path.join(args.save_dir, "training_args.bin"))
    
    if args.do_eval:
        results = {}
        checkpoint_file = os.path.join(args.save_dir, args.checkpoint_name)
        global_step = checkpoint_file.split('-')[-1]  
        model = LongformerForTokenClassification.from_pretrained(os.path.join(args.save_dir, args.checkpoint_name))
        model.to(args.device)
        result = evaluate(model, tokenizer, args, prefix=global_step)
        result = {k + "_{}".format(global_step): v for k, v in result.items()}
        results.update(result)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=4,
        type=int
    )
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float
    )
    parser.add_argument(
        '--learning_rate',
        default=5e-5
    )
    parser.add_argument(
        '--adam_epsilon',
        default=1e-8
    )
    parser.add_argument(
        '--warmup_steps',
        default=1000
    )
    parser.add_argument(
        '--train_epochs',
        default=2
    )
    parser.add_argument(
        '--seed',
        default=42
    )
    parser.add_argument(
        '--max_grad_norm',
        default=1.0,
        help='used at gradient clip'
    )
    parser.add_argument(
        '--logging_steps',
        default=500
    )
    parser.add_argument(
        '--save_steps',
        default=1000
    )
    parser.add_argument(
        '--save_dir',
        required=True,
        help='checkpoint, binary file saving dir'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='text file(ex. results.txt) saving dir'
    )
    parser.add_argument(
        '--do_train',
        default=1,
        type=int
    )
    parser.add_argument(
        '--do_eval',
        default=1,
        type=int
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        help='name of checkpoint to load model'
    )
    args = parser.parse_args()

    main(args)