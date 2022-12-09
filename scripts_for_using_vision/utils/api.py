import numpy as np

from .dialogue import main_function as dialogue_main_function
from .metadata import main_function as metadata_main_function
from .scene import Scene


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
    
    def given_scene_get_all_obj_info(self, scene_name: str):
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

    def dial_data_returner(self, len_history=2):
        """
        returns dial data including coref and belief state. this does not make any kind of a file, but just returns a list, because it's more flexible.
        len_history: length of history (1 user uttr + 1 system uttr = 1 history). 
        Total utterance for each time is therefore 2*len_history + 1 (current user utter)

        [
            {  // start of a dialogue
                domain: dialogue's domain
                dialogues: [
                    {'context': history_1 + user_uttr_1 as list, 'context_with_obj': context with system's mentioned objs, 'belief': belief_1 dict},
                    {'context': history_2 + user_uttr_2 as list, 'context_with_obj': context with system's mentioned objs, 'belief': belief_2 dict},
                    ...
                ]
                scene_objects: {
                    scene_idx_1: {
                    0: 0-th object's meta info, in dictionary format
                    1: 1-th object's meta info, in dictionary format
                    ...
                    },
                    scene_idx_2: {
                    0: 0-th object's meta info, in dictionary format
                    1: 1-th object's meta info, in dictionary format
                    ...
                    },
                    ...
                }
            }  // end of a dialogue
            ...
        ]
        """
        dialogue_data = []
        for dialogue in self.all_dialogues.dialogue_list:
            dialogue_dict = {'domain': dialogue.domain, 'scene_objects': dict(), 'dialogues': []}
            scene_ids = dialogue.scene_ids
            for k, scene_id in scene_ids.items():
                dialogue_dict['scene_objects'][int(k)] = dict()
                scene = Scene.from_json(scene_id)
                scene_objs_info = self.given_scene_get_all_obj_info(scene_id)
                for obj_info in scene_objs_info:
                    # TODO: convert world to camera-relative position, considering camera's position and orientation (direction vec -> ouler angle -> subtract displacement then apply rotation matrix)
                    obj_info_dict = {**vars(obj_info['meta']), **vars(obj_info['obj'])}  # order of 'meta' and 'obj' matters
                    # convert object's world position to camera-relative position
                    # stackoverflow.com/questions/21622956/how-to-convert-direction-vector-to-euler-angles
                    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
                    
                    # Actually, only position from camera matters... for directional utterance, using bbox is better
                    obj_world_pos = obj_info['obj'].position
                    camera_pos = scene.camera_object.camera
                    distance_from_camera = np.linalg.norm(np.array(obj_world_pos) - np.array(camera_pos))
                    obj_info_dict['distance'] = distance_from_camera
                    dialogue_dict['scene_objects'][int(k)][obj_info['obj'].index] = obj_info_dict
            
            for idx, single_turn in enumerate(dialogue.single_dialogue_list):
                single_turn_dict = dict()
                belief = single_turn.transcript_annotated
                belief_dict = {'act': ":".join([belief.act.dialogue_act, belief.act.activity]), 'slot_values': belief.act_attributes.slot_values,
                               'request_slots': belief.act_attributes.request_slots, 'objects': belief.act_attributes.objects}
                single_turn_dict['belief'] = belief_dict
                context_list = []
                context_with_obj_list = []
                context_list.insert(0, 'USER : ' + single_turn.transcript)
                context_with_obj_list.insert(0, 'USER : ' + single_turn.transcript)
                for i in range(1, len_history+1):
                    if idx - i >= 0:
                        one_history = 'USER : ' + dialogue.single_dialogue_list[idx-i].transcript + ' SYSTEM : ' + dialogue.single_dialogue_list[idx-i].system_transcript
                        context_list.insert(0, one_history)
                        obj = ', '.join(list(map(str,dialogue.single_dialogue_list[idx-i].system_transcript_annotated.act_attributes.objects)))
                        one_history_with_obj = 'USER : ' + dialogue.single_dialogue_list[idx-i].transcript + ' SYSTEM : ' + dialogue.single_dialogue_list[idx-i].system_transcript + ' <SOM> ' + obj + ' <EOM>'
                        context_with_obj_list.insert(0, one_history_with_obj)
                    else:
                        break
                single_turn_dict['context'] = context_list
                single_turn_dict['context_with_obj'] = context_with_obj_list
                dialogue_dict['dialogues'].append(single_turn_dict)

            dialogue_data.append(dialogue_dict)

        return dialogue_data
    
    # def item_tokens_attrs(item2id, ) 


if __name__ == "__main__":
    prompt_api = PromptAPI('dev')
    metas = prompt_api.given_scene_get_all_obj_info('m_cloth_store_1416238_woman_3_8')
    meta_dict = {**vars(metas[0]['meta']), **vars(metas[0]['obj'])}
    print(metas)


    