import re
import json
import argparse

from typing import Dict, List


def parse_flattened_result(to_parse):
    dialog_act_regex = re.compile(r'([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\>')    
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= (\[(.*)\]|[^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []
    # Parse
    if "=> Belief State : " not in to_parse:
        splits = ['', to_parse.strip()]
    else:
        splits = to_parse.strip().split("=> Belief State : ")
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split("<EOB>")

        if len(splits) == 2:
            # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
            to_parse = splits[0].strip()

            for dialog_act in dialog_act_regex.finditer(to_parse):
                d = {
                    "act": dialog_act.group(1),
                    "slots": [],
                    "request_slots": [],
                    "objects": [],
                }

                for slot in slot_regex.finditer(dialog_act.group(2)):
                    d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])

                for request_slot in request_regex.finditer(dialog_act.group(3)):
                    d["request_slots"].append(request_slot.group(1).strip())

                for object_id in object_regex.finditer(dialog_act.group(4)):
                    str_object_id = object_id.group(1).strip()

                    try:
                        # Object ID should always be <int>.
                        int_object_id = int(str_object_id)
                        d["objects"].append(int_object_id)
                    except:
                        pass

                if d != {}:
                    belief.append(d)
    return belief


def format_for_dst(predictions: List[str]) -> List[Dict]:
    '''
        Formats model predictions for subtask 2, 3.

        NOTE: This follows the format given by the baseline.

        Args:
            predictions <List[str]>: predictions outputted by model
        Returns:
            submission <List[Dict]>: submission format
    '''
    submission = list()
    for pred in predictions:
        submission.append(
            parse_flattened_result(pred)
        )
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction', type=str,
        default=None, help="file to convert (line-by-line *.txt format)"
    )
    parser.add_argument(
        '--output', type=str,
        default='dstc10-simmc-teststd-pred-subtask-3.json',
        help="json output path"
    )
    args = parser.parse_args()

    with open(args.prediction, 'r') as f:
        prediction = list(f.readlines())

    submission = format_for_dst(prediction)

    with open(args.output, 'w') as f:
        json.dump(submission, f)