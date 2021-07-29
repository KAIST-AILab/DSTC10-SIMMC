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
TEMPLATE_TARGET = ("{context} {START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"


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


def format_dialog(
    dialog,
    oov=None,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
):

    prev_asst_uttr = None
    prev_turn = None
    lst_context = []

    for turn in dialog[FIELDNAME_DIALOG]:
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
                                               use_belief_states)

    # If a new output path for special tokens is given, we track new OOVs
    oov = None
    if output_path_special_tokens:
        oov = set()

    _formatter = partial(format_dialog,
                         oov=oov,
                         len_context=len_context,
                         use_multimodal_contexts=use_multimodal_contexts,
                         use_belief_states=use_belief_states)
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


def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([str(o) for o in object_ids])
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
