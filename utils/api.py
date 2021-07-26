import attr
from typing import Dict, List
from attr.validators import instance_of
from util import *


class PromptAPI:
    def __init__(self, dial_split=None):
        assert dial_split in {'dial_train', 'dial_dev', 'dial_devtest', 'dial_test', None}
        if dial_split:
            self.dial_split = dial_split
            self.all_dialogues = load_pickle(pickle_type=dial_split)
        self.fashion_meta = load_pickle(pickle_type="fashion_meta")
        self.furniture_meta = load_pickle(pickle_type="furniture_meta")

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

    def attribute_search(self, scene_name: str, attr_type, attr_range):
        scene = Scene.from_json(scene_name)
        if 'cloth' in scene_name:
            domain_metadata = self.fashion_meta
        elif 'wayfair' in scene_name:
            domain_metadata = self.furniture_meta
        else:
            raise ValueError("scene_name should contain either word 'cloth' or 'wayfair'")







# prompt_api = PromptAPI()
# metadata = prompt_api.given_scene_objid_get_meta('cloth_store_1_1_1', obj_unique_id=0)
# print('print metadata', metadata)
