import json

furniture_metadata_dir = '../../../../../data/furniture_prefab_metadata_all.json'

with open(furniture_metadata_dir, 'r') as f:
    furniture_metadata = json.load(f)

colors = set()
types = set()

for k, v in furniture_metadata.items():
    colors.add(v['color'])
    types.add(v['type'])

with open('color_furniture.txt', 'wt') as f:
    color_strs = []
    for i, v in enumerate(list(colors)):
        color_strs.append(f'{v} {i}')
    f.write('\n'.join(color_strs))

with open('type_furniture.txt', 'wt') as f:
    type_strs = []
    for i, v in enumerate(list(types)):
        type_strs.append(f'{v} {i}')
    f.write('\n'.join(type_strs))