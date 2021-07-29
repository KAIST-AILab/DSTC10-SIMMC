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
import collections
import json
import os
from functools import partial
from itertools import chain

from gpt2_dst.utils.convert import represent_visual_objects

from . import api

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
START_OF_META = "<META>"
END_OF_META = "</META>"

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} "
TEMPLATE_TARGET = ("{context} {START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")

TEMPLATE_PREDICT_USE_META = "{context} {metainfo} {START_BELIEF_STATE} "
TEMPLATE_TARGET_USE_META = (
    "{context} {metainfo} {START_BELIEF_STATE} {belief_state} "
    "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"

prompt_api = api.PromptAPI()


def _build_special_tokens(use_multimodal_contexts=True,
                          use_belief_states=True,
                          use_metainfo=False):
    special_tokens = {"eos_token": END_OF_SENTENCE}
    additional_special_tokens = []
    if use_belief_states:
        additional_special_tokens.append(END_OF_BELIEF)
    else:
        additional_special_tokens.append(START_OF_RESPONSE)
    if use_multimodal_contexts:
        additional_special_tokens.append(START_OF_MULTIMODAL_CONTEXTS)
        additional_special_tokens.append(END_OF_MULTIMODAL_CONTEXTS)
    if use_metainfo:
        additional_special_tokens.extend([START_OF_META, END_OF_META])
    special_tokens["additional_special_tokens"] = additional_special_tokens

    special_tokens = {"eos_token": END_OF_SENTENCE}
    return special_tokens


def format_dialog(dialog,
                  oov=None,
                  len_context=2,
                  use_multimodal_contexts=True,
                  use_belief_states=True,
                  use_scene_ids=False,
                  use_metainfo=False):
    scene_ids = dialog["scene_ids"]
    prev_asst_uttr = None
    prev_turn = None
    lst_context = []
    if use_scene_ids:
        scene_str = "".join([k + ":" + v + ", " for k, v in scene_ids.items()])
        scene_str = scene_str.rsplit(',', 1)[0]  # remove tailing ","
        predict = "Scene ) {scene_str}"
        target = "Scene ) {scene_str}"
        yield predict, target

    for turn_idx, turn in enumerate(dialog[FIELDNAME_DIALOG]):
        user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
        user_belief = turn[FIELDNAME_BELIEF_STATE]
        asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

        # Format main input context
        context = ""
        if prev_asst_uttr:
            context += f"System : {prev_asst_uttr} "
            if use_multimodal_contexts:
                # Add multimodal contexts
                visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                    "act_attributes"]["objects"]
                context += represent_visual_objects(visual_objects) + " "

        context += f"User : {user_uttr}"
        prev_asst_uttr = asst_uttr
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-len_context:])

        if use_metainfo:
            # get scene_id
            od = collections.OrderedDict(
                sorted(scene_ids.items(), key=lambda t: int(t[0])))
            od_list = list(od.items())
            idx_scene = [(int(idx), scene_id) for idx, scene_id in od_list]
            this_turn_scene_id = ""
            for i in range(len(idx_scene)):
                if idx_scene[i][0] <= turn_idx:
                    this_turn_scene_id = idx_scene[i][1]

            # get objects' meta info in the scene
            scene_objs = prompt_api.given_scene_get_all_meta(
                this_turn_scene_id)
            scene_obj_dict = {}

            for obj in scene_objs:
                if 'wayfair' not in this_turn_scene_id:
                    # unique_id, bbox and position, up down left right are omitted
                    scene_obj_dict[obj['obj'].index] = {
                        'asset_type': obj['meta'].asset_type,
                        'customer_review': obj['meta'].customer_review,
                        'available_sizes': obj['meta'].available_sizes,
                        'color': obj['meta'].color,
                        'pattern': obj['meta'].pattern,
                        'brand': obj['meta'].brand,
                        'sleeve_length': obj['meta'].sleeve_length,
                        'type': obj['meta'].type,
                        'price': obj['meta'].price,
                        'size': obj['meta'].size
                    }
                else:
                    scene_obj_dict[obj['obj'].index] = {
                        'brand': obj['meta'].brand,
                        'color': obj['meta'].color,
                        'customer_review': obj['meta'].customer_review,
                        'materials': obj['meta'].materials,
                        'price': obj['meta'].price,
                        'type': obj['meta'].type
                    }

            obj_meta_str = START_OF_META + ' '
            for key, value in scene_obj_dict.items():
                obj_meta_str += str(key) + ':' + ' '.join(
                    [str(v) for v in value.values()]) + ' '
            obj_meta_str += END_OF_META

        # Format belief state
        if use_belief_states:
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

            # Track OOVs
            if oov is not None:
                oov.add(user_belief["act"])
                for slot_name in user_belief["act_attributes"]["slot_values"]:
                    oov.add(str(slot_name))

            str_belief_state = " ".join(belief_state)
            # Format the main input
            if use_metainfo:
                predict = TEMPLATE_PREDICT_USE_META.format(
                    context=context,
                    metainfo=obj_meta_str,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                )
            else:
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
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
        yield predict, target


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    use_scene_ids=False,
    use_metainfo=False,
    input_path_special_tokens="",
    output_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    if input_path_special_tokens:
        with open(input_path_special_tokens, "r") as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = _build_special_tokens(use_multimodal_contexts,
                                               use_belief_states, use_metainfo)

    # If a new output path for special tokens is given, we track new OOVs
    oov = None
    if output_path_special_tokens:
        oov = set()

    _formatter = partial(format_dialog,
                         oov=oov,
                         len_context=len_context,
                         use_multimodal_contexts=use_multimodal_contexts,
                         use_belief_states=use_belief_states,
                         use_scene_ids=use_scene_ids,
                         use_metainfo=use_metainfo)
    predicts, targets = zip(*chain.from_iterable(map(_formatter, data)))

    directory = os.path.dirname(output_path_predict)
    os.makedirs(directory, exist_ok=True)

    directory = os.path.dirname(output_path_target)
    os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        f_predict.write("\n".join(predicts))

    with open(output_path_target, "w") as f_target:
        f_target.write("\n".join(targets))

    if output_path_special_tokens:
        directory = os.path.dirname(output_path_special_tokens)
        os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)


if __name__ == '__main__':
    input_path_json = '/home/haeju/Dev/dstc/dstc10/ours/data/simmc2_dials_dstc10_train.json'
    output_path_predict = '/home/haeju/Dev/dstc/dstc10/ours/model/mm_dst/bart_dst/data_custom/simmc2_dials_dstc10_train_predict.txt'
    output_path_target = '/home/haeju/Dev/dstc/dstc10/ours/model/mm_dst/bart_dst/data_custom/simmc2_dials_dstc10_train_target.txt'
    output_path_special_tokens = '/home/haeju/Dev/dstc/dstc10/ours/model/mm_dst/bart_dst/data_custom/simmc_special_tokens.json'
    len_context = 2
    input_path_special_tokens = ""
    use_multimodal_contexts = True
    use_belief_states = True
    use_scene_ids = True
    use_metainfo = True

    convert_json_to_flattened(
        input_path_json,
        output_path_predict,
        output_path_target,
        input_path_special_tokens=input_path_special_tokens,
        output_path_special_tokens=output_path_special_tokens,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts,
        use_belief_states=use_belief_states,
        use_scene_ids=use_scene_ids,
        use_metainfo=use_metainfo)
