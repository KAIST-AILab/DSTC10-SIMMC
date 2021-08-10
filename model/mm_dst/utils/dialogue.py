import json
from os.path import join
from typing import Dict, List

import attr
from attr.validators import instance_of

from .util import find_data_dir

DIALOGUE_ACTS = ("INFORM", "CONFIRM", "REQUEST", "ASK")
ACTIVITIES = ("GET", "DISAMBIGUATE", "REFINE", "ADD_TO_CART", "COMPARE")

DATA_DIR = find_data_dir('DSTC10-SIMMC')  # give root folder name of simmc2 as argument. Ex) find_data_dir('DSTC10-SIMMC')

@attr.s
class Action:
    dialogue_act: str = attr.ib(
        converter=lambda x: str(x).upper()
    )
    activity: str = attr.ib(
        converter=lambda x: str(x).upper()
    )

    @staticmethod
    def check_in(attribute, value, listing):
        """Universal checker that validates if value is in the given list."""
        if value not in listing:
            raise ValueError("{} must be one of {}, but received {}.".format(attribute.name, listing, value))
    
    @dialogue_act.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, DIALOGUE_ACTS)

    @activity.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, ACTIVITIES)

@attr.s
class ActionAttributes:
    slot_values: Dict = attr.ib()
    request_slots: List = attr.ib()
    objects: List[int] = attr.ib()


@attr.s
class TranscriptAnnotation:
    act: str = attr.ib()
    act_attributes: ActionAttributes = attr.ib()

    @classmethod
    def annotation_filler(cls, act: str, act_attributes):
        dialogue_act, activity = act.split(':')
        act_args = {
            'dialogue_act': dialogue_act,
            'activity': activity
        }
        act_filled = Action(**act_args)
        act_attributes_filled = ActionAttributes(**act_attributes)
        args = {
            'act': act_filled,
            'act_attributes': act_attributes_filled
        }
        return cls(**args)

@attr.s
class SingleDialogueTurn:
    turn_idx: int = attr.ib()
    system_transcript: str = attr.ib()
    system_transcript_annotated: TranscriptAnnotation = attr.ib()
    transcript: str = attr.ib()
    transcript_annotated: TranscriptAnnotation = attr.ib()
    disambiguation_label = attr.ib(default=None)

    @classmethod
    def single_dialogue_filler(cls, turn_idx, transcript, transcript_annotated,
                               system_transcript, system_transcript_annotated, disambiguation_label=None):

        transcript_annotated_filled = TranscriptAnnotation.annotation_filler(**transcript_annotated)
        system_transcript_annotated_filled = TranscriptAnnotation.annotation_filler(**system_transcript_annotated)
        args = {
            'turn_idx': turn_idx,
            'transcript': transcript,
            'transcript_annotated': transcript_annotated_filled,
            'system_transcript': system_transcript,
            'system_transcript_annotated': system_transcript_annotated_filled,
            'disambiguation_label': disambiguation_label
        }
        return cls(**args)

@attr.s
class Dialogue:
    dialogue_idx: int = attr.ib(
        validator=instance_of(int)
    )
    domain: str = attr.ib(
        converter=lambda x: str(x).lower(),
        validator=instance_of(str)
    )
    mentioned_object_ids: List[int] = attr.ib(
        converter=lambda x: [int(_) for _ in x],
        validator=instance_of(list)
    )
    scene_ids: Dict[int, str] = attr.ib(
        converter=lambda x: {int(k): str(v) for k,v in x.items()},
        validator=instance_of(dict)
    )
    single_dialogue_list: List[SingleDialogueTurn] = attr.ib()

    @domain.validator
    def check(self, attribute, value):
        if value not in ("fashion", "furniture"):
            raise ValueError("Domain must either be fashion or furniture.")

    @classmethod
    def dialogue_filler(cls, dialogue, dialogue_idx, domain, mentioned_object_ids, scene_ids):
        single_dialogue_list = list()
        for idx, single_dialogue in enumerate(dialogue):
            single_dialogue_list.append(SingleDialogueTurn.single_dialogue_filler(**single_dialogue))
        args = {
            'single_dialogue_list': single_dialogue_list,
            'dialogue_idx': dialogue_idx,
            'domain': domain,
            'mentioned_object_ids': mentioned_object_ids,
            'scene_ids': scene_ids
        }
        dialogue_idx
        return cls(**args)


@attr.s
class AllDialogues:

    dialogue_list: List[Dialogue] = attr.ib()
    dialogue_split: str = attr.ib()
    dialogue_domain: str = attr.ib()

    @classmethod
    def from_json(cls, dialogue_name: str):
        dialogue_json = json.load(open(join(DATA_DIR, "{}.json".format(dialogue_name))))
        dialogue_list = list()
        dialogue_split = dialogue_json['split']
        dialogue_domain = dialogue_json['domain']
        for idx, dialogue in enumerate(dialogue_json['dialogue_data']):
            dialogue_args = {
                'dialogue': dialogue['dialogue'],
                'dialogue_idx': dialogue['dialogue_idx'],
                'domain': dialogue['domain'],
                'mentioned_object_ids': dialogue['mentioned_object_ids'],
                'scene_ids': dialogue['scene_ids'],
            }
            dialogue_list.append(Dialogue.dialogue_filler(**dialogue_args))

        args = {
            'dialogue_list': dialogue_list,
            'dialogue_split': dialogue_split,
            'dialogue_domain': dialogue_domain,
        }
        return cls(**args)


def main_function(dial_split="train"):
    assert dial_split in {"train", "dev", "devtest", "test"}, "Give the right split name: should be one of train, dev, devtest, test"
    all_dialogues = AllDialogues.from_json(f'simmc2_dials_dstc10_{dial_split}')
    return all_dialogues


if __name__ == "__main__":
    all_dials = main_function(dial_split='dev')
    print(all_dials.dialogue_list[:2])
