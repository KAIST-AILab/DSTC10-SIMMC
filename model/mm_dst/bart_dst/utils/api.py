import attr
import copy

from typing import Dict, List
from attr.validators import instance_of
from pickle_loader import load_pickle

from dialogue import main_function as dialogue_main_function
from metadata import main_function as metadata_main_function
from scene import Scene


class PromptAPI:
    def __init__(self, dial_split=None):
        assert dial_split in {'train', 'dev', 'devtest', 'test', None}
        if dial_split:
            self.dial_split = dial_split
            self.all_dialogues = dialogue_main_function(dial_split=dial_split)
        self.fashion_meta, self.furniture_meta = metadata_main_function()

    def given_scene_objid_get_meta(self, scene_name: str, obj_unique_id=None, obj_index=None):
        assert (obj_unique_id is None or obj_index is None) and (not (obj_unique_id is None and obj_index is None)), \
            "either only one of obj_unique_id and obj_index should have value"
        scene = Scene.from_json(scene_name)
        # print('scene', scene)
        if 'cloth' in scene_name:
            domain_metadata = self.fashion_meta
        elif 'wayfair' in scene_name:
            domain_metadata = self.furniture_meta
        else:
            raise ValueError("scene_name should contain either word 'cloth' or 'wayfair'")

        if obj_unique_id is not None:
            for obj in scene.scene_object:
                if obj.unique_id == int(obj_unique_id):
                    for meta in domain_metadata:
                        if meta.name == obj.prefab_path:
                            return meta  # instance of {Fashion|Furniture}Metadata

        if obj_index is not None:
            for obj in scene.scene_object:
                if obj.index == int(obj_index):
                    for meta in domain_metadata:
                        if meta.name == obj.prefab_path:
                            return meta  # instance of {Fashion|Furniture}Metadata
    
    def given_scene_get_all_meta(self, scene_name: str):
        scene = Scene.from_json(scene_name)
        if 'cloth' in scene_name:
            domain_metadata = self.fashion_meta
        elif 'wayfair' in scene_name:
            domain_metadata = self.furniture_meta
        else:
            raise ValueError("scene_name should contain either word 'cloth' or 'wayfair'")

        scene_obj_meta_list = []

        for obj in scene.scene_object:
            for meta in domain_metadata:
                if meta.name == obj.prefab_path:
                    scene_obj_meta_list.append({'obj': obj, 'meta': meta})

        return scene_obj_meta_list

    def given_belief_get_obj(self, scene_name: str, belief_state: str):
        """
        aim to get proper object, given (generated) belief state
        belief_state: string between "=> Belief State : " and "<EOB>"
        """
        scene_obj_meta_list = self.given_scene_get_all_meta(scene_name)
        belief_state = belief_state.strip()
        act = belief_state.split('[')[0]
        slot_string = belief_state.split('[')[1].split(']')[0]
        slot_list = [s.replace(' ','') for s in slot_string.split(',')]
        slot = dict()
        for k_v in slot_list:
            if k_v != '' and k_v != ' ':
                print("k_v",k_v)
                k, v = k_v.split('=')
                slot[k] = v
        request_slot = belief_state.split('(')[1].split(')')[0].replace(' ', '').split(',')
        objects = list(map(int, belief_state.split('<')[1].split('>')[0].replace(' ', '').split(',')))
        



# prompt_api = PromptAPI('train')
# metadata = prompt_api.given_scene_objid_get_meta('cloth_store_1_1_1', obj_unique_id=0)
# print('print metadata', metadata)
# metas = prompt_api.given_scene_get_all_meta('m_cloth_store_1416238_woman_3_8')
