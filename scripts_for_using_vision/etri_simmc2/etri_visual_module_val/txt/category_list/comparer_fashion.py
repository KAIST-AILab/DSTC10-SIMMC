import json

with open('assetType.txt', 'r') as f:
    assetTypes = [line.strip().rsplit(' ', 1)[0] for line in f.readlines()]

with open('new_color.txt', 'r') as f:
    colors = [line.rstrip().rsplit(' ', 1)[0] for line in f.readlines()]

with open('pattern.txt', 'r') as f:
    patterns = [line.rstrip().rsplit(' ', 1)[0] for line in f.readlines()]

with open('sleeveLength.txt', 'r') as f:
    sleeveLengths = [line.rstrip().rsplit(' ', 1)[0] for line in f.readlines()]

with open('type.txt', 'r') as f:
    types = [line.rstrip().rsplit(' ', 1)[0] for line in f.readlines()]   

fashion_metadata_dir = '../../../../../data/fashion_prefab_metadata_all.json'


with open(fashion_metadata_dir, 'r') as f:
    fashion_metadata = json.load(f)

missing_colors = []  # only color is missing
for k, v in fashion_metadata.items():
    if v['assetType'] not in assetTypes:
        print(f"assetType {v['assetType']} is missing!!")
    if v['color'] not in colors:
        print(f"color {v['color']} is missing!!")
        missing_colors.append(v['color'])
    if v['pattern'] not in patterns:
        print(f"pattern {v['pattern']} is missing!!")
    if v['sleeveLength'] not in sleeveLengths:
        print(f"sleeveLength {v['sleeveLength']} is missing!!")
    if v['type'] not in types:
        print(f"type {v['type']} is missing!!")

print(list(set(missing_colors)))