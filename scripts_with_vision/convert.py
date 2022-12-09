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
import os
import json
import argparse
import collections

from functools import partial
from itertools import chain

import imagesize
import numpy as np

from utils import api
from utils.metadata import (
    FASHION_SIZES, FASHION_AVAILABLE_SIZES,
    FASHION_BRAND, FASHION_PRICE,
    FASHION_CUSTOMER_REVIEW, FURNITURE_BRAND,
    FURNITURE_PRICE, FURNITURE_CUSTOMER_RATING
)

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
START_OF_OBJ_TOKEN = "<SOO>"
END_OF_OBJ_TOKEN = "<EOO>"
OBJ_START = "<OBJ>"
OBJ_PREVI = "<PREVIOBJ>"
DET_START = "<DET>"
NO_COREF = "<NOCOREF>"

available_sizes2st = {
    'XS': '<A>',
    'S': '<B>',
    'M': '<C>',
    'L': '<D>',
    'XL': '<E>',
    'XXL': '<F>' 
}

# If we use each object token as special token
NUM_FASHION_ITEMS = 288
NUM_FURNITURE_ITEMS = 57
MAX_NUM_OBJ_IN_SCENE = 200

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} "
TEMPLATE_TARGET = ("{context} {START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")
TEMPLATE_TARGET_FINAL = ("{context} {START_BELIEF_STATE} {belief_state} "
                         "{END_OF_BELIEF} {response} {END_OF_SENTENCE} {disambig_str}")

TEMPLATE_PREDICT_USE_OBJVEC = "{context} {objvec} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_USE_OBJVEC_DET = "{context} {det} {objvec} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_OBJVEC_FIRST = "{objvec} {context} {START_BELIEF_STATE} "
TEMPLATE_PREDICT_OBJVEC_FIRST_DET = "{det} {objvec} {context} {START_BELIEF_STATE} "
TEMPLATE_FINAL = "{context} {START_BELIEF_STATE} {det} {objvec}"  # seg: 2 0 1 0 1, ... 


# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"


prompt_api = api.PromptAPI()
def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def represent_visual_objects_special_token(object_ids, for_belief_state=False):
    # Stringify visual objects (JSON)
    str_objects = ", ".join(["<"+str(o)+">" for o in object_ids])
    if for_belief_state:
        return str_objects
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def arrange_det(scene_json_folder, scene_id):
    det_arrange_list = []
    scene_id_for_img = scene_id[2:] if scene_id.startswith('m_') else scene_id 
    if scene_id_for_img in det_info:
        det_scene = det_info[scene_id_for_img]
        img_w = det_scene['width']
        img_h = det_scene['height']
        
        for det in det_scene['det']:
            x1 = det['rect']['x1']
            y1 = det['rect']['y1']
            x2 = det['rect']['x2']
            y2 = det['rect']['y2']
            label = det['label']
            pos_str = '{}{}[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(DET_START, label, x1/img_w -0.5, y1/img_h -0.5, x2/img_w -0.5, y2/img_h -0.5, (x2-x1)*(y2-y1)/(img_w*img_h))
            det_arrange_list.append(pos_str)
        return ''.join(det_arrange_list)
    else:
        return ''
    

def arrange_object_special_tokens(scene_json_folder, image_folder, scene_ids, object_item2id, insert_bbox_coords):
    arrange_list = []
    scene_loaded_list = []
    obj_dict_possibly_duplicated = dict()
    for scene_id_idx, scene_id in enumerate(scene_ids):
        with open(os.path.join(scene_json_folder, f"{scene_id}_scene.json"), 'r') as f_in:
            scene = json.load(f_in)
        scene_loaded_list.append(scene)
        for obj in scene['scenes'][0]['objects']: 
            obj_dict_possibly_duplicated[obj['index']] = scene_id_idx
    
    num_scene = len(scene_ids)
    for scene_id_idx, scene_id in enumerate(scene_ids):
        scene = scene_loaded_list[scene_id_idx]
        bbox_id = scene_id[2:] if scene_id.startswith('m_') else scene_id 
        with open(os.path.join(scene_json_folder, f"{bbox_id}_bbox.json"), 'r') as f_in:
            bbox = json.load(f_in)
        camera_position = []; camera_dir_vec = []
        for bbox_item in bbox['Items']:
            if bbox_item['name'] == 'camera':
                camera_position = np.array(bbox_item['position'])
            if bbox_item['name'] == 'camera_forward':
                camera_dir_vec = np.array(bbox_item['position'])

        if insert_bbox_coords:
            largest_z_value = 0
            for obj in scene['scenes'][0]['objects']:
                position = np.array(obj['position'])
                obj_displacement = position - camera_position
                theta = np.dot(obj_displacement, camera_dir_vec) / (np.linalg.norm(obj_displacement)*np.linalg.norm(camera_dir_vec))
                largest_z_value = max(np.linalg.norm(obj_displacement) * np.cos(theta), largest_z_value)
        for obj in scene['scenes'][0]['objects']:
            assert obj['index'] in obj_dict_possibly_duplicated, "SOMETHING IS MISSING!"
            if scene_id_idx == obj_dict_possibly_duplicated[obj['index']]:
                if insert_bbox_coords:
                    position = np.array(obj['position'])
                    obj_displacement = position - camera_position
                    theta = np.dot(obj_displacement, camera_dir_vec) / (np.linalg.norm(obj_displacement)*np.linalg.norm(camera_dir_vec))
                    z_value = np.linalg.norm(obj_displacement) * np.cos(theta)
                    
                    # image name 
                    image_id = None
                    if "m" in scene_id[0]: image_id = scene_id[2:]
                    else: image_id = scene_id
                    image_file_name = os.path.join(image_folder, image_id+".png")
                    if os.path.exists(image_file_name):
                        img_w, img_h = imagesize.get(image_file_name)
                        x1, y1, h, w = obj['bbox']
                        x2, y2 = x1 + w, y1 + h
                        pos_str = '[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(x1/img_w -0.5, y1/img_h -0.5, x2/img_w -0.5, y2/img_h -0.5, (x2-x1)*(y2-y1)/(img_w*img_h), z_value/largest_z_value)
                    else:
                        print(f'{scene_id} is not present in img_size!!!')
                        pos_str = '[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(0.0, 0.0, 0.0, 0.0, 0.0, z_value/largest_z_value)
                else:
                    pos_str = ''

                if (num_scene != 1) and (scene_id_idx == 0): 
                    arrange_list.append(OBJ_PREVI + "<" + str(obj['index']) + ">" + pos_str + object_item2id[obj['prefab_path']])
                else: 
                    arrange_list.append(OBJ_START + "<" + str(obj['index']) + ">" + pos_str + object_item2id[obj['prefab_path']])
    return ''.join(arrange_list)


def get_scene_id(scene_ids, this_turn, so_far=False):
    """
        scene_ids: dict, whose keys are dialogue turn idx and values are scene_id
        this_turn: int, of current dialogue turn idx
    """
    od = collections.OrderedDict(
        sorted(scene_ids.items(), key=lambda t: int(t[0])))
    od_list = list(od.items())
    idx_scene = [(int(idx), scene_id) for idx, scene_id in od_list]
    
    if so_far:
        return list([x[1] for x in idx_scene if x[0] <= this_turn])

    for i in range(len(idx_scene)):
        if idx_scene[i][0] <= this_turn:
            this_turn_scene_id = idx_scene[i][1]
    return this_turn_scene_id


def format_dialog(dialog,
                  len_context=2,
                  use_multimodal_contexts=True,
                  use_belief_states=True,
                  object_item2id=None,
                  scene_json_folder='',
                  image_folder='',
                  insert_bbox_coords=True,
                  revert=False,
                  with_target=True):
    scene_ids = dialog["scene_ids"]
    dialog_idx = dialog['dialogue_idx']
    prev_asst_uttr = None
    prev_turn = None
    lst_context = []

    for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):

        user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()        
        
        if with_target:
            user_belief = turn[FIELDNAME_BELIEF_STATE]
        
        if "system_transcript" in turn:
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
        else:
            # print(f"Diag ID : {dialog_idx}, turn_id :{turn_idx}")
            asst_uttr = ''
        
        this_turn_scene_id = get_scene_id(scene_ids, turn_idx)
        scene_ids_so_far = get_scene_id(scene_ids, turn_idx, so_far=True)
        # Format main input context
        context = ""
        if prev_asst_uttr:
            context += f"System : {prev_asst_uttr} "
            if use_multimodal_contexts:
                # Add multimodal contexts
                visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                    "act_attributes"]["objects"]

                if object_item2id is not None:
                    context += represent_visual_objects_special_token(visual_objects, for_belief_state=False) + " "
                else:
                    context += represent_visual_objects(visual_objects) + " "

        context += f"User : {user_uttr}"
        prev_asst_uttr = asst_uttr
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])
                
        if object_item2id is not None:
            object_token_arranged = arrange_object_special_tokens(scene_json_folder, image_folder, scene_ids_so_far, object_item2id, insert_bbox_coords)
            obj_token_str = START_OF_OBJ_TOKEN + NO_COREF + object_token_arranged + END_OF_OBJ_TOKEN
        
        # Format belief state
        if use_belief_states:
            if with_target:
                if object_item2id is not None:
                    belief_state = []
                    act = user_belief["act"].strip()
                    slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}" if k!='availableSizes' else '{} = {}'.format(k.strip(), str([available_sizes2st[x] for x in v]).replace("'", "").strip()) 
                    # slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}" if k!='availableSizes' else f'{k.strip()} = {str([available_sizes2st[x] for x in v]).replace("\'", "").strip()}'
                                            for k, v in user_belief["act_attributes"]
                                            ["slot_values"].items())
                    request_slots = ", ".join(
                        user_belief["act_attributes"]["request_slots"])
                    objects_str = represent_visual_objects_special_token(user_belief["act_attributes"]["objects"], for_belief_state=True)
                    # for bs_per_frame in user_belief:
                    str_belief_state_per_frame = (
                        f"{act} [ {slot_values} ] ({request_slots}) < {objects_str} >")
                    belief_state.append(str_belief_state_per_frame)
                else:
                    belief_state = []
                    act = user_belief["act"].strip()
                    slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}"
                                            for k, v in user_belief["act_attributes"]
                                            ["slot_values"].items())
                    request_slots = ", ".join(
                        user_belief["act_attributes"]["request_slots"])
                    objects = ", ".join(
                        map(str, user_belief["act_attributes"]["objects"]))
                    # for bs_per_frame in user_belief:
                    str_belief_state_per_frame = (
                        f"{act} [ {slot_values} ] ({request_slots}) < {objects} >")
                    belief_state.append(str_belief_state_per_frame)

                str_belief_state = " ".join(belief_state)
            
            # Format the main input
            if object_item2id is not None: 

                if not revert:
                    predict = TEMPLATE_PREDICT_USE_OBJVEC.format(
                        context=context,
                        objvec=obj_token_str,
                        START_BELIEF_STATE=START_BELIEF_STATE
                    )
                else:
                    predict = TEMPLATE_PREDICT_OBJVEC_FIRST.format(
                        objvec=obj_token_str,
                        context=context,
                        START_BELIEF_STATE=START_BELIEF_STATE
                    )
            else:
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                )

            if with_target:
                # Format the main output
                target = TEMPLATE_TARGET.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                    belief_state=str_belief_state,
                    END_OF_BELIEF=END_OF_BELIEF,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                )
            else: target=""
        else:
            # Format the main input
            predict = TEMPLATE_PREDICT_NOBELIEF.format(
                context=context, START_OF_RESPONSE=START_OF_RESPONSE)

            if with_target:
                # Format the main output
                target = TEMPLATE_TARGET_NOBELIEF.format(
                    context=context,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
            else: target=""
        yield predict, target


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    object_special_token_item2id="",
    scene_json_folder='',
    image_folder='',
    insert_bbox_coords=True,
    revert=False,
    with_target=True
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """
    if object_special_token_item2id:
        with open(object_special_token_item2id, 'r') as f_in:
            object_item2id = json.load(f_in)
        use_object_special_token = True
    else:
        use_object_special_token = False

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    _formatter = partial(format_dialog,
                         len_context=len_context,
                         use_multimodal_contexts=use_multimodal_contexts,
                         use_belief_states=use_belief_states,
                         object_item2id=object_item2id,
                         scene_json_folder=scene_json_folder,
                         image_folder=image_folder,
                         insert_bbox_coords=insert_bbox_coords,
                         revert=revert,
                         with_target=with_target)
    predicts, targets = zip(*chain.from_iterable(map(_formatter, data)))

    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        f_predict.write("\n".join(predicts))

    if with_target:
        with open(output_path_target, "w") as f_target:
            f_target.write("\n".join(targets))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_json', type=str)
    parser.add_argument('--output_path_predict', type=str)
    parser.add_argument('--output_path_target', type=str)
    parser.add_argument('--len_context', default=2, type=int)
    parser.add_argument('--use_multimodal_contexts', type=int, default=1)
    parser.add_argument('--use_belief_states', type=int, default=1)
    parser.add_argument('--object_special_token_item2id', type=str)
    parser.add_argument('--scene_json_folder', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--insert_bbox_coords', type=int, default=1)
    parser.add_argument('--revert', type=int, default=0)
    parser.add_argument('--with_target', type=int, default=1)
    args = parser.parse_args()
    
    convert_json_to_flattened(
        args.input_path_json,
        args.output_path_predict,
        args.output_path_target,
        len_context=args.len_context,
        use_multimodal_contexts=args.use_multimodal_contexts,
        use_belief_states=args.use_belief_states,
        object_special_token_item2id=args.object_special_token_item2id,
        scene_json_folder=args.scene_json_folder,
        image_folder=args.image_folder,
        insert_bbox_coords=args.insert_bbox_coords,
        revert=args.revert,
        with_target=args.with_target)