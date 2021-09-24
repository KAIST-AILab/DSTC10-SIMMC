#!/usr/bin/env python3
import json
import os
import re
import collections
import numpy as np
import copy
import ipdb
from tqdm import tqdm 
from PIL import Image
# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"


# Template for Bart formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
# START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_ENCODER_INPUT_IDS = "{context}"
TEMPLATE_DECODER_INPUT_IDS = ("{belief_state} {END_OF_BELIEF} {response} {END_OF_SENTENCE}") # starting with </s> by tokenizer
data_root_dir = "/home/yschoi/data"
# Scene json 
scene_data_dir = "/home/yschoi/data/jsons"
# Object Meta Data
fashion_meta_file = "/home/yschoi/data/fashion_prefab_metadata_all.json"
furniture_meta_file = "/home/yschoi/data/furniture_prefab_metadata_all.json"
# Object Unique Index
item2idx = json.load(open("item2id.json"))

idx2item = {}
for k, v in item2idx.items():
    idx2item[v] = k

fashion_meta_data = json.load(open(fashion_meta_file, "r"))

samples = []

def format_dialog(dialog, len_context=2, use_multimodal_contexts=True, use_belief_states=True, with_scene_id=True):
    scene_ids = dialog["scene_ids"]
    prev_turn_scene_id = None
    accum_objs = {}    
    prev_asst_uttr = None
    prev_turn = None
    lst_context = []

    meta_data = {}
    for scene_id in scene_ids.values():
        scene_id += "_scene.json"
        if not os.path.exists(os.path.join(scene_data_dir, scene_id)): ipdb.set_trace()
        scene_items =json.load(open(os.path.join(scene_data_dir, scene_id)))["scenes"][0]["objects"]

        objs_in_scene = []
        bbox_list = [] 
        abs_position = []
        visual_objs_dicts = {}
        
        # objs meta data
        for obj in scene_items:
            obj_name = obj["prefab_path"]
            objs_in_scene.append(item2idx[obj_name])
            visual_objs_dicts[obj["index"]] = item2idx[obj_name]
            bbox_list.append(obj["bbox"])
            abs_position.append(obj["position"])
        
        meta_data[scene_id] = {
            "objs_in_scene" : objs_in_scene,
            "bbox" : bbox_list,
            "abs_position" : abs_position,
            "visual_objs_dicts" : visual_objs_dicts,
        }
        
    for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
        user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
        user_belief = turn[FIELDNAME_BELIEF_STATE]
        asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
        asst_belief = turn[FIELDNAME_SYSTEM_STATE]

        ##############################################################################
        if with_scene_id:
            od = collections.OrderedDict(
                sorted(scene_ids.items(), key=lambda t: int(t[0])))
            od_list = list(od.items())
            idx_scene = [(int(idx), scene_id) for idx, scene_id in od_list]
            this_turn_scene_id = ""
            for i in range(len(idx_scene)):
                if idx_scene[i][0] <= turn_idx:
                    this_turn_scene_id = idx_scene[i][1]

        this_turn_image_id_with_folder = None
        this_turn_image_id = None
        if "m" in this_turn_scene_id[0]: this_turn_image_id = this_turn_scene_id[2:]
        else: this_turn_image_id = this_turn_scene_id
        
        # IMAGE DATA CHECK
        this_turn_image_id_with_folder = os.path.join(data_root_dir, "images", this_turn_image_id+".png")
        if not os.path.exists(this_turn_image_id_with_folder): 
            print(this_turn_image_id)
            height, width = 900, 1800
        else : 
            width, height = Image.open(this_turn_image_id_with_folder).size

        # SCENE JSON DATA CHECK
        this_turn_scene_id +="_scene.json"
        
        # Accum Data
        if prev_turn_scene_id != this_turn_scene_id:
            for k, v in meta_data[this_turn_scene_id]['visual_objs_dicts'].items(): # k: object index v: unique item idx
                if k in accum_objs:
                    del accum_objs[k]
                accum_objs[k] = v

        # Format main input context
        context = ""
        context += f"User : {user_uttr} "
        context += f"System : {asst_uttr}"

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])

        # Previous Info
        prev_turn_scene_id = this_turn_scene_id
        encoder_input = TEMPLATE_ENCODER_INPUT_IDS.format(
            context=context,
        )
        samples.append({
            "encoder_input" : encoder_input + " <EOS>",
            # "available_objs" : list(accum_objs.values()), # object unide id <@1234>
            # "objs_inds" : list(accum_objs.keys()), # object scene_id
            # "bbox" : meta_data[this_turn_scene_id]["bbox"],
            # "image_size" : [height, width]
        })

def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([ str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_json",
        type=str,
        default='/home/yschoi/data/simmc2_dials_dstc10_train.json'
    )
    parser.add_argument(
        "--output_path_json",
        type=str,
        default="/home/yschoi/SIMMC2/data/retrieval/train.json"    
    ) 
    parser.add_argument(
        "--num_aug_tag",
        type=int,
        default=0
    ) 
    parser.add_argument(
        "--with_meta_info",
        type=bool,
        default=False
    ) 
    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path_json = args.output_path_json

    dataset = json.load(open(input_path_json))["dialogue_data"]
    for diag in tqdm(dataset, "Converting"):
        # if diag["domain"] == "furniture":
        #     continue 
        format_dialog(diag)
    
    for _ in range(args.num_aug_tag):
        for diag in tqdm(dataset, "Converting"):
            format_dialog_with_aug_tags(diag)

    data_format = {}
    for idx, sample in enumerate(samples):
        data_format[idx] = sample
    
    json.dump(data_format, open(output_path_json, "w"), indent=4)
    