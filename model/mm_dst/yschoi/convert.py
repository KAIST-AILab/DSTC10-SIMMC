#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import json
import os
import re
from functools import partial
from itertools import chain
import collections
import csv
import numpy as np
import copy
import ipdb

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} "

TEMPLATE_PREDICT_WITH_ITEM = "{item} {context} {END_OF_SENTENCE}"
TEMPLATE_TARGET = ("{context} {START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")
TEMPLATE_LABEL = ("{START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")
TEMPLATE_BELIEF = ("{belief_state} {END_OF_BELIEF}")

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"

slot_types = set()
def _build_special_tokens(use_multimodal_contexts=True,
                          use_belief_states=True):
    special_tokens = {"eos_token": END_OF_SENTENCE}

    additional_special_tokens = []
    if use_belief_states:
        additional_special_tokens.append(END_OF_BELIEF)
    else:
        additional_special_tokens.append(START_OF_RESPONSE)
    if use_multimodal_contexts:
        additional_special_tokens.append(START_OF_MULTIMODAL_CONTEXTS)
        additional_special_tokens.append(END_OF_MULTIMODAL_CONTEXTS)

    special_tokens["additional_special_tokens"] = additional_special_tokens
    return special_tokens

data_root_dir = "/ext/coco_dataset/simmc2/data"
# Image data
folder_list = ["simmc2_scene_images_dstc10_public_part1", "simmc2_scene_images_dstc10_public_part2"]

# Scene json 
scene_data_dir = "/ext/coco_dataset/simmc2/data/public"

# Object Meta Data
fashion_meta_file = "/ext/coco_dataset/simmc2/data/fashion_prefab_metadata_all.json"
furniture_meta_file = "/ext/coco_dataset/simmc2/data/furniture_prefab_metadata_all.json"

# Object Unique Index
item2idx = json.load(open("/ext/dstc10/yschoi/item2id.json"))
check_dialogue = []

def format_dialog(
    dialog,
    oov=None,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    with_scene_id=True,
):
    scene_ids = dialog["scene_ids"]
    prev_asst_uttr = None
    prev_turn = None
    lst_context = []
    prev_turn_scene_id = None
    accum_objs = {}

    ############################################################################################
    meta_data = {}
    for scene_id in scene_ids.values():
        scene_id += "_scene.json"
        if not os.path.exists(os.path.join(scene_data_dir, scene_id)): ipdb.set_trace()
        scene_items =json.load(open(os.path.join(scene_data_dir, scene_id)))["scenes"][0]["objects"]

        objs_in_scene = []
        bbox_list = [] 
        abs_position = []
        visual_objs_dicts = {}

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
            "visual_objs_dicts" : visual_objs_dicts
        }
        
    for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
        user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
        user_belief = turn[FIELDNAME_BELIEF_STATE]
        asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

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
        for folder in folder_list:
            if os.path.exists(os.path.join(data_root_dir, folder, this_turn_image_id+".png")):
                this_turn_image_id_with_folder = os.path.join(folder, this_turn_image_id+".png")
                break
        if this_turn_image_id_with_folder is None: print(this_turn_image_id)
        

        # SCENE JSON DATA CHECK
        this_turn_scene_id +="_scene.json"
        
        # Accum Data
        if prev_turn_scene_id != this_turn_scene_id:
            for k, v in meta_data[this_turn_scene_id]['visual_objs_dicts'].items(): # k: object index v: unique item idx
                accum_objs[k] = v
        #################################################################################

        # Format main input context
        context = ""
        if prev_asst_uttr:
            context += f"System : {prev_asst_uttr} "
            if use_multimodal_contexts:
                # Add multimodal contexts
                visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                    "act_attributes"]["objects"] 
                try :
                    visual_objects_unique_idx = [accum_objs[idx] for idx in visual_objects]
                except :
                    ipdb.set_trace()
                    check_dialogue.append(dialog["dialogue_idx"])
                context += represent_visual_objects(visual_objects_unique_idx) + " "
        context += f"User : {user_uttr}"
        prev_asst_uttr = asst_uttr
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])

        # Previous Info
        prev_turn_scene_id = this_turn_scene_id

        # Format belief state

        # binary classification label
        binary_class_objects = {}
        for k in accum_objs.keys():
            binary_class_objects[k] = 0
        
        if use_belief_states:
            belief_state = []
            act = user_belief["act"].strip()
            for k, v in user_belief["act_attributes"]["slot_values"].items():
                slot_types.add(k)

            slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}"
                                    for k, v in user_belief["act_attributes"]
                                    ["slot_values"].items())
            if "Object" in slot_values: ipdb.set_trace()
            
            request_slots = ", ".join(
                user_belief["act_attributes"]["request_slots"])

            mentioned_objects = [ accum_objs[idx] for idx in user_belief["act_attributes"]["objects"]]
            avaiable_objects = list(accum_objs.values())
            for idx in user_belief["act_attributes"]["objects"]:
                binary_class_objects[idx] = 1

            # objects = ", ".join(
            #     map(str, user_belief["act_attributes"]["objects"]))
            objects = ", ".join(mentioned_objects)
            # for bs_per_frame in user_belief:
            str_belief_state_per_frame = (
                f"{act} [ {slot_values} ] ({request_slots}) < {objects} >")
            belief_state.append(str_belief_state_per_frame)

            # Track OOVs
            if oov is not None:
                oov.add(user_belief["act"])
                for slot_name in user_belief["act_attributes"]["slot_values"]:
                    oov.add(str(slot_name))
        
            str_belief_state = " ".join(belief_state)

            # Format the main input
            predict = TEMPLATE_PREDICT.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE,
            )

            objects = ", ".join(avaiable_objects)
            predict_with_item = TEMPLATE_PREDICT_WITH_ITEM.format(
                item="{ " + objects +" }",
                context=context,
                END_OF_SENTENCE=END_OF_SENTENCE
            )
            # Format the main output
            target = TEMPLATE_TARGET.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE,
                belief_state=str_belief_state,
                END_OF_BELIEF=END_OF_BELIEF,
                response=asst_uttr,
                END_OF_SENTENCE=END_OF_SENTENCE,
            )

            # Format the label
            label = TEMPLATE_LABEL.format(
                START_BELIEF_STATE=START_BELIEF_STATE,
                belief_state=str_belief_state,
                END_OF_BELIEF=END_OF_BELIEF,
                response=asst_uttr,
                END_OF_SENTENCE=END_OF_SENTENCE,
            )

            belief = TEMPLATE_BELIEF.format(
                belief_state=str_belief_state,
                END_OF_BELIEF=END_OF_BELIEF
            )
        else:
            # Format the main input
            predict = TEMPLATE_PREDICT_NOBELIEF.format(
                context=context, START_OF_RESPONSE=START_OF_RESPONSE)

            # Format the main output
            target = TEMPLATE_TARGET_NOBELIEF.format(
                context=context,
                response=asst_uttr,
                END_OF_SENTENCE=END_OF_SENTENCE,
                START_OF_RESPONSE=START_OF_RESPONSE,
            )

        yield predict, predict_with_item, target, belief, this_turn_scene_id, this_turn_image_id_with_folder,  \
            meta_data[this_turn_scene_id]["objs_in_scene"], meta_data[this_turn_scene_id]["bbox"], \
                copy.deepcopy(accum_objs), copy.deepcopy(mentioned_objects), copy.deepcopy(binary_class_objects)

def convert_json_to_flattened(
    input_path_json,
    output_path_json,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """
    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]
    # If a new output path for special tokens is given, we track new OOVs
    oov = None
    _formatter = partial(format_dialog,
                         oov=oov,
                         len_context=len_context,
                         use_multimodal_contexts=use_multimodal_contexts,
                         use_belief_states=use_belief_states)
    predicts, predicts_with_item, targets, belief, scene_idxes, images, objs_in_scene, bbox_list, \
        accum_objs, mentioned_objs, mentioned_objs_labels = zip(*chain.from_iterable(map(_formatter, data)))

    print("="*50)
    print(slot_types)
    
    directory = os.path.dirname(output_path_json)
    os.makedirs(directory, exist_ok=True)

    with open(output_path_json, "w") as file:
        data_dict = {}
        data_idx = 0
        for img, scene, predict, target, label, objects, bbox , accum_obj, mentioned_obj, mentioned_objs_label \
            in zip(images, scene_idxes, predicts_with_item, targets, belief, objs_in_scene, bbox_list, accum_objs, mentioned_objs, mentioned_objs_labels):
            if img is None: 
                print("None image!") 
                continue
            # if "wayfair" in img:
            #     continue
            if scene is None:
                print("None Scene!")
                continue
            if (len(predict) > 0 and not predict.isspace()):
                asset = {}
                asset['image'] = img
                asset['predict'] = predict
                asset['belief'] = label
                asset['objects'] = objects #[f"<@{o}>" for o in objects]
                asset['bbox'] = bbox # list
                asset['accum_obj'] = list(accum_obj.values()) #[f"<@{o}>" for o in accum_obj.values()]
                asset['mentioned_obj'] = mentioned_obj
                asset['coref_label'] = [int(value) for value in mentioned_objs_label.values()]
                data_dict[data_idx] = asset
                data_idx += 1
            else:
                print("None predict data!!")
        json.dump(data_dict, file, indent=4)

def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([ o for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"

def parse_flattened_results_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    dialog_act_regex = re.compile(
        r"([\w:?.?]*)  *\[([^\]]*)\] *\(([^\]]*)\) *\<([^\]]*)\>")
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= ([^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []

    # Parse
    splits = to_parse.strip().split(START_BELIEF_STATE)
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split(END_OF_BELIEF)

        if len(splits) == 2:
            # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
            to_parse = splits[0].strip()

            for dialog_act in dialog_act_regex.finditer(to_parse):
                act, slots, requests, objects = dialog_act.groups()
                d = {
                    "act": act,
                    "slots": [],
                    "request_slots": [],
                    "objects": [],
                }

                for slot in slot_regex.finditer(slots):
                    d["slots"].append([slot[1].strip(), slot[2].strip()])

                for request_slot in request_regex.finditer(requests):
                    d["request_slots"].append(request_slot[1].strip())

                for object_id in object_regex.finditer(objects):
                    d["objects"].append(object_id[1].strip())

                if d != {}:
                    belief.append(d)

    return belief

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_json",
        type=str,
        default='/ext/coco_dataset/simmc2/data/simmc2_dials_dstc10_train.json'
    )
    parser.add_argument(
        "--output_path_json",
        type=str,
        default="/ext/dstc10/output.json"    
    )
    parser.add_argument("--len_context", type=int, default=2)
    parser.add_argument(
        "--use_multimodal_contexts",
        help="determine whether to use the multimodal contexts each turn",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no_belief_states",
        dest="use_belief_states",
        action="store_false",
        default=True,
        help="determine whether to use belief state for each turn",
    )
    
    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path_json = args.output_path_json
    len_context = args.len_context
    use_multimodal_contexts = bool(args.use_multimodal_contexts)

    # DEBUG:
    print("Belief states: {}".format(args.use_belief_states))

    # Convert the data into GPT-2 friendly format
    convert_json_to_flattened(
        input_path_json,
        output_path_json,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts,
        use_belief_states=args.use_belief_states,
    )
