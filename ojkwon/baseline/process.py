import re
import os
import json

from typing import Dict, List, Any, Set
from pathlib import Path
from functools import partial
from itertools import chain


PAD_TOKEN = "<PAD>"
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_SOURCE = "{context} {START_BELIEF_STATE} "
TEMPLATE_TARGET = ("{context} {START_BELIEF_STATE} {belief_state} "
                   "{END_OF_BELIEF} {response} {END_OF_SENTENCE}")
TEMPLATE_SOURCE_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"

def batch_remove_pad_token(sentences: List[str]) -> List[str]:
    return [
        s.split(PAD_TOKEN)[0].strip() for s in sentences
    ]
    

def get_special_tokens(
    use_belief_states: bool=True,
    use_multimodal_contexts: bool=True
):
    special_tokens = {"eos_token": END_OF_SENTENCE, "pad_token": PAD_TOKEN}
    additional_special_tokens = list()
    if use_belief_states:
        additional_special_tokens.append(END_OF_BELIEF)
    else:
        additional_special_tokens.append(START_OF_RESPONSE)
    if use_multimodal_contexts:
        additional_special_tokens.append(START_OF_MULTIMODAL_CONTEXTS)
        additional_special_tokens.append(END_OF_MULTIMODAL_CONTEXTS)
    special_tokens['additional_special_tokens'] = additional_special_tokens
    return special_tokens


def format_dialog(
    dialog: Dict,
    oov: Set,
    context_length: int=2,
    use_multimodal_contexts: bool=True,
    use_belief_states: bool=True
):
    prev_sys_attr = None
    prev_turn = None
    lst_context = list()

    for turn in dialog['dialogue']:
        usr_uttr = turn['transcript'].replace("\n", " ").strip()
        usr_belief = turn['transcript_annotated']
        sys_attr = turn['system_transcript'].replace("\n", " ").strip()

        # Format main input context
        context = ""
        if prev_sys_attr:
            context += f"System : {prev_sys_attr} "
            if use_multimodal_contexts:
                # Add multimodal contexts
                visual_objects = prev_turn['system_transcript_annotated'][
                    "act_attributes"]["objects"]
                objects = ", ".join([str(o) for o in visual_objects])
                context += f"{START_OF_MULTIMODAL_CONTEXTS} {objects} {END_OF_MULTIMODAL_CONTEXTS} "

        context += f"User : {usr_uttr}"
        prev_sys_attr = sys_attr
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-context_length:])

        # Format belief state
        if use_belief_states:
            belief_state = []
            act = usr_belief["act"].strip()
            slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}"
                                    for k, v in usr_belief["act_attributes"]
                                    ["slot_values"].items())
            request_slots = ", ".join(
                usr_belief["act_attributes"]["request_slots"])
            objects = ", ".join(
                map(str, usr_belief["act_attributes"]["objects"]))
            # for bs_per_frame in usr_belief:
            str_belief_state_per_frame = (
                f"{act} [ {slot_values} ] ({request_slots}) < {objects} >")
            belief_state.append(str_belief_state_per_frame)

            # Track OOVs
            if oov is not None:
                oov.add(usr_belief["act"])
                for slot_name in usr_belief["act_attributes"]["slot_values"]:
                    oov.add(str(slot_name))

            str_belief_state = " ".join(belief_state)

            # Format the main input
            source = TEMPLATE_SOURCE.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE
            )

            # Format the main output
            target = TEMPLATE_TARGET.format(
                context=context,
                START_BELIEF_STATE=START_BELIEF_STATE,
                belief_state=str_belief_state,
                END_OF_BELIEF=END_OF_BELIEF,
                response=sys_attr,
                END_OF_SENTENCE=END_OF_SENTENCE,
            )
        else:
            # Format the main input
            source = TEMPLATE_TARGET_NOBELIEF.format(
                context=context, START_OF_RESPONSE=START_OF_RESPONSE)

            # Format the main output
            target = TEMPLATE_TARGET_NOBELIEF.format(
                context=context,
                response=sys_attr,
                END_OF_SENTENCE=END_OF_SENTENCE,
                START_OF_RESPONSE=START_OF_RESPONSE,
            )
        yield source, target


def convert_json_to_flattened(
    data: List[Dict[str, Any]],
    output_path: str,
    context_length: int=2,
    use_multimodal_contexts: bool=True,
    use_belief_states: bool=True,
):
    '''
        Preprocesses dialog dataset given in json format to flattened texts.

        Args:
            data <List[Dict[str, Any]]>: list of dialogues

    '''
    oov = set()
    _formatter = partial(
        format_dialog,
        oov=oov,
        context_length=context_length,
        use_multimodal_contexts=use_multimodal_contexts,
        use_belief_states=use_belief_states
    )
    sources, targets = zip(*chain.from_iterable(map(_formatter, data)))

    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    # Output into text files
    to_dump = {"source": sources, "target": targets}
    with open(output_path, "w") as f:
        json.dump(to_dump, f)
    return oov


def parse_flattened_result(to_parse: str) -> List[Dict[str, Any]]:
    '''
        Parse line-by-line raw text data to belief state dictionary.
        Used for evaluation of generated text sequence.

        Args:
            to_parse <str>: a single text line
        
        Returns:
            parsed <List[Dict[str, Any]]>: parsed result in json format

            e.g.
            [
                {
                    'act': <str>  # e.g. 'DA:REQUEST',
                    'slots': [
                        <str> slot_name,
                        <str> slot_value
                    ]
                }, ...  # End of a frame
            ]  # End of a dialog
    '''
    dialog_act_regex = re.compile(
        r"([\w:?.?]*)  *\[([^\]]*)\] *\(([^\]]*)\) *\<([^\]]*)\>"
    )
    slot_regex = re.compile(
        r"([A-Za-z0-9_.-:]*)  *= ([^,]*)"
    )
    request_regex = re.compile(
        r"([A-Za-z0-9_.-:]+)"
    )
    object_regex = re.compile(
        r"([A-Za-z0-9]+)"
    )
    belief = list()

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
