import re
import os
import json

from typing import Dict, List, Any, Set
from functools import partial
from itertools import chain


START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

USR_UTTR_TOKEN = "<USR>"
SYS_UTTR_TOKEN = "<SYS>"

TEMPLATE_SOURCE = "{context}"
TEMPLATE_TARGET = "{belief_state} {END_OF_BELIEF} {response}"


def clean_tokens(
    sentences: List[str],
    pad_token: str,
    bos_token: str,
    eos_token: str
) -> List[str]:
    return [
        s.replace(pad_token, '')
         .strip(eos_token)
         .strip(bos_token)
         .strip(eos_token[:-1])
         .strip(bos_token[1:])
         .strip() for s in sentences
    ]


def parse_response(generated: str):
    splits = generated.split(END_OF_BELIEF, 1)
    try:
        result = (splits[0].strip(), splits[1].strip())
    except:
        result = (generated, '')
    return result


def get_special_tokens(generate_sys_attr: bool=False):
    special_tokens = dict()
    additional_special_tokens = [USR_UTTR_TOKEN, SYS_UTTR_TOKEN]
    additional_special_tokens.append(END_OF_BELIEF)
    if generate_sys_attr:
        additional_special_tokens.append(START_OF_RESPONSE)
    additional_special_tokens.append(START_OF_MULTIMODAL_CONTEXTS)
    additional_special_tokens.append(END_OF_MULTIMODAL_CONTEXTS)
    special_tokens['additional_special_tokens'] = additional_special_tokens
    return special_tokens   


def format_dialog(
    dialog: Dict,
    oov: Set,
    context_length: int=2,
    generate_sys_attr=False
):
    prev_sys_attr = None
    prev_turn = None
    lst_context = list()
    
    dialog_id = dialog['dialogue_idx']
    scene_ids = dialog['scene_ids']

    curr_scene_id = scene_ids['0']
    for turn in dialog['dialogue']:
        turn_id = turn['turn_idx']
        curr_scene_id = scene_ids.get(str(turn_id), curr_scene_id)
        if curr_scene_id is not None:
            prev_scene_id = curr_scene_id
        else:
            curr_scene_id = prev_scene_id
        usr_uttr = turn['transcript'].replace("\n", " ").strip()
        usr_belief = turn['transcript_annotated']
        sys_attr = turn['system_transcript'].replace("\n", " ").strip()
        disamb_label = turn.get('disambiguation_label', -100)

        # Format main input context
        context = ""
        visual_objects = list()
        if prev_sys_attr:
            context += f"{SYS_UTTR_TOKEN} {prev_sys_attr} "
            # MM contexts
            visual_objects = prev_turn['system_transcript_annotated'][
                "act_attributes"]["objects"]
            objects = ", ".join([str(o) for o in visual_objects])
            context += f"{START_OF_MULTIMODAL_CONTEXTS} {objects} {END_OF_MULTIMODAL_CONTEXTS} "

        context += f"{USR_UTTR_TOKEN} {usr_uttr}"
        prev_sys_attr = sys_attr    
        prev_turn = turn

        # Concat with previous contexts
        lst_context.append(context)
        context = " ".join(lst_context[-context_length:])

        # Format belief state
        belief_state = []
        act = usr_belief["act"].strip()
        # In the form of "<type> = shirt, <pattern> = striped"
        slot_values = ", ".join(f"{k.strip()} = {str(v).strip()}"
                                for k, v in usr_belief["act_attributes"]
                                ["slot_values"].items())
        request_slots = ", ".join(usr_belief["act_attributes"]["request_slots"])
        # bs_objects = usr_belief['act_attributes']['objects']
        bs_objects = ", ".join(map(str, usr_belief["act_attributes"]["objects"]))
        str_belief_state_per_frame = f"{act} [ {slot_values} ] ({request_slots}) < >"
        str_belief_state_per_frame = f"{act} [ {slot_values} ] ({request_slots}) < {bs_objects} >"
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
        )
        yield (
            dialog_id,
            turn_id,
            source,
            disamb_label,
        )


def convert_json_to_flattened(
    dialogs: List[Dict[str, Any]],
    output_path: str,
    context_length: int=2,
    generate_sys_attr: bool=False
):
    oov = set()
    _formatter = partial(
        format_dialog,
        oov=oov,
        context_length=context_length,
        generate_sys_attr=generate_sys_attr
    )
    (
        dialog_id,
        turn_id,
        sources,
        disamb_label,
    ) = zip(*chain.from_iterable(map(_formatter, dialogs)))
    
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    # Output into text files
    to_dump = list()
    for di, ti, src, dl in zip(dialog_id, turn_id, sources, disamb_label):
        to_dump.append({
            'dialog_id': di,
            'turn_id': ti,
            'source': src,
            'disambiguation_label': dl,
        })
    with open(output_path, "w") as f:
        json.dump(to_dump, f, indent=4)
    return oov
