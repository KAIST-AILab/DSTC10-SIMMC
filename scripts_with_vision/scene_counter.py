import os
import json


data_folder = '../data'

with open(os.path.join(data_folder, 'simmc2_dials_dstc10_devtest.json'), 'r') as f:
    devtest_json = json.load(f)

with open(os.path.join(data_folder, 'simmc2_dials_dstc10_train.json'), 'r') as f:
    train_json = json.load(f)


json_files = [devtest_json, train_json]
scene_ids_devtest = dict()
scene_ids_train = dict()

for json in json_files:
    dialogs = json['dialogue_data']
    for dial in dialogs:
        if json['split'] == 'devtest':
            for scene in dial['scene_ids'].values():
                if scene in scene_ids_devtest:
                    scene_ids_devtest[scene] += 1
                else:
                    scene_ids_devtest[scene] = 1

        
        if json['split'] == 'train':
            for scene in dial['scene_ids'].values():
                if scene in scene_ids_train:
                    scene_ids_train[scene] += 1
                else:
                    scene_ids_train[scene] = 1

# print(scene_ids_train)

scene_ids_devtest = {k: v for k, v in sorted(scene_ids_devtest.items(), key=lambda item: item[1])}      
scene_ids_train = {k: v for k, v in sorted(scene_ids_train.items(), key=lambda item: item[1])}             

for i, k in enumerate(scene_ids_devtest.keys()):
    # print(f'devtest) {k}: {scene_ids_devtest[k]}')
    pass
print(i)

for i, k in enumerate(scene_ids_train.keys()):
    # print(f'train) {k}: {scene_ids_train[k]}')
    pass
print(i)

for k, v in scene_ids_devtest.items():
    if k in scene_ids_train:
        print(k)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

print(intersection(list(scene_ids_devtest.keys()), list(scene_ids_train.keys())))
