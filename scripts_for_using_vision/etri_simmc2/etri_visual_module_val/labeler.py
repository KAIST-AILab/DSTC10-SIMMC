import os
import json

data_folder = '../../../data'
images_cropped_folder = './images_cropped'

with open(os.path.join(data_folder, 'simmc2_dials_dstc10_devtest.json'), 'r') as f:
    devtest_json = json.load(f)

with open(os.path.join(data_folder, 'simmc2_dials_dstc10_train.json'), 'r') as f:
    train_json = json.load(f)

with open(os.path.join(data_folder, 'fashion_prefab_metadata_all.json'), 'r') as f:
    fashion_prefab = json.load(f)

with open(os.path.join(data_folder, 'furniture_prefab_metadata_all.json'), 'r') as f:
    furniture_prefab = json.load(f)

with open(os.path.join('txt/category_list', 'assetType_fashion.txt'), 'r') as f:
    assetType_lines = f.readlines()
    assetType_dict = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in assetType_lines}
with open(os.path.join('txt/category_list', 'color_fashion.txt'), 'r') as f:
    color_lines = f.readlines()
    color_dict = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in color_lines}
with open(os.path.join('txt/category_list', 'pattern_fashion.txt'), 'r') as f:
    pattern_lines = f.readlines()
    pattern_dict = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in pattern_lines}
with open(os.path.join('txt/category_list', 'sleeveLength_fashion.txt'), 'r') as f:
    sleeveLength_lines = f.readlines()
    sleeveLength_dict = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in sleeveLength_lines}
with open(os.path.join('txt/category_list', 'type_fashion.txt'), 'r') as f:
    type_lines = f.readlines()
    type_dict = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in type_lines}

with open(os.path.join('txt/category_list', 'color_furniture.txt'), 'r') as f:
    color_lines = f.readlines()
    color_dict_furniture = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in color_lines}
with open(os.path.join('txt/category_list', 'type_furniture.txt'), 'r') as f:
    type_lines = f.readlines()
    type_dict_furniture = {line.rstrip().rsplit(' ', 1)[0]: int(line.rstrip().rsplit(' ', 1)[1]) for line in type_lines}


json_files = [devtest_json, train_json]
devtest_scenes = set()
train_scenes = set()

for json_file in json_files:
    dialogs = json_file['dialogue_data']
    for dial in dialogs:
        if json_file['split'] == 'devtest':
            for scene in dial['scene_ids'].values():
                devtest_scenes.add(scene)

        if json_file['split'] == 'train':
            for scene in dial['scene_ids'].values():
                train_scenes.add(scene)


train_label_lines = []
for scene in list(train_scenes):
    with open(os.path.join(data_folder, 'jsons', f'{scene}_scene.json')) as f:
        scene_json = json.load(f)
        scene_cropped_folder = os.path.join(images_cropped_folder, scene)
        scene_cropped_list = os.listdir(scene_cropped_folder)
        scene_cropped_dict = {int(x.split('_', 1)[0]): x for x in scene_cropped_list}

        for obj in scene_json['scenes'][0]['objects']:
            if obj['index'] in scene_cropped_dict:  # sometimes no cropped image for object
                path_string = f"{scene}/{scene_cropped_dict[obj['index']]} "
                if 'wayfair' not in scene:  # which means fashion
                    fashion_item_meta = fashion_prefab[obj['prefab_path']]
                    a = assetType_dict[fashion_item_meta['assetType']]
                    c = color_dict[fashion_item_meta['color']]
                    p = pattern_dict[fashion_item_meta['pattern']]
                    s = sleeveLength_dict[fashion_item_meta['sleeveLength']]
                    t = type_dict[fashion_item_meta['type']]
                    path_string += f'{a} {c} {p} {s} {t}'
                else:
                    furniture_item_meta = furniture_prefab[obj['prefab_path']]
                    c = color_dict_furniture[furniture_item_meta['color']]
                    t = type_dict_furniture[furniture_item_meta['type']]
                    path_string += f'{c} {t}'
                train_label_lines.append(path_string)

with open('./txt/train_labels.txt', 'wt') as f:
    f.write('\n'.join(train_label_lines))


devtest_label_lines = []
for scene in list(devtest_scenes):
    with open(os.path.join(data_folder, 'jsons', f'{scene}_scene.json'), 'r') as f:
        scene_json = json.load(f)
        scene_cropped_folder = os.path.join(images_cropped_folder, scene)
        scene_cropped_list = os.listdir(scene_cropped_folder)
        scene_cropped_dict = {int(x.split('_', 1)[0]): x for x in scene_cropped_list}

        for obj in scene_json['scenes'][0]['objects']:
            if obj['index'] in scene_cropped_dict:  # sometimes no cropped image for object
                path_string = f"{scene}/{scene_cropped_dict[obj['index']]} "
                if 'wayfair' not in scene:  # which means fashion
                    fashion_item_meta = fashion_prefab[obj['prefab_path']]
                    a = assetType_dict[fashion_item_meta['assetType']]
                    c = color_dict[fashion_item_meta['color']]
                    p = pattern_dict[fashion_item_meta['pattern']]
                    s = sleeveLength_dict[fashion_item_meta['sleeveLength']]
                    t = type_dict[fashion_item_meta['type']]
                    path_string += f'{a} {c} {p} {s} {t}'
                else:
                    furniture_item_meta = furniture_prefab[obj['prefab_path']]
                    c = color_dict_furniture[furniture_item_meta['color']]
                    t = type_dict_furniture[furniture_item_meta['type']]
                    path_string += f'{c} {t}'
                devtest_label_lines.append(path_string)

with open('./txt/devtest_labels.txt', 'wt') as f:
    f.write('\n'.join(devtest_label_lines))


