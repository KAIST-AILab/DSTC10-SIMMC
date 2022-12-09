import json
import os 

with open('./item2id.json', 'r') as f_in:
    item2id = json.load(f_in)

item2id_sep = dict()
for k, v in item2id.items():
    item2id_sep[k] = [v, v.replace('@', '#')]

with open('./item2id_sep.json', 'w') as f:
    json.dump(item2id_sep, f)


with open('../data_object_special/simmc_special_tokens.json', 'r') as f_in:
    simmc_special_tokens = json.load(f_in)
print("simmc_special_tokens['additional_special_tokens']", simmc_special_tokens['additional_special_tokens'])
fashion_nonvis = [f"<#1{i:03}>" for i in range(288)]
furniture_nonvis = [f"<#2{i:03}>" for i in range(57)]

simmc_special_tokens['additional_special_tokens'].extend(fashion_nonvis)
simmc_special_tokens['additional_special_tokens'].extend(furniture_nonvis)

with open('../data_object_special_sep/simmc_special_tokens.json', 'w') as f_out:
    json.dump(simmc_special_tokens, f_out)

splits = ['train', 'dev', 'devtest']
for split in splits:
    filename = f'../data_object_special/simmc2_dials_dstc10_{split}_predict.txt'
    targetfilename = f'../data_object_special_sep/simmc2_dials_dstc10_{split}_predict.txt'
    newlines = []
    with open(filename, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if (len(line) > 0 and not line.isspace()):
                i = 0
                while line[i-5:i] != 'te : ':

                # for i in range(len(line)):
                    if line[i] == '@' and line[i-1] == '<':
                        line = line[:i+6] + f'<#{line[i+1:i+5]}>' + line[i+6:]

                    i += 1
                newlines.append(line)

    with open(targetfilename, 'w') as f:
        f.write('\n'.join(newlines))

acts = ["REQUEST:ADD_TO_CART", "REQUEST:COMPARE", "INFORM:DISAMBIGUATE", 
        "ASK:GET", "INFORM:GET", "REQUEST:GET", "INFORM:REFINE"]
for split in splits:
    filename = f'../data_object_special/simmc2_dials_dstc10_{split}_target.txt'
    targetfilename = f'../data_object_special_sep/simmc2_dials_dstc10_{split}_target.txt'
    newlines = []
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f.read().splitlines()):
            if (len(line) > 0 and not line.isspace()):
                for act in acts:
                    if act in line:
                        newlines.append(line.replace(act, f'<{act}>'))
                        break
        assert i == len(newlines), f"WRONG! {i}, {len(newlines)}"
            
    with open(targetfilename, 'w') as f:
        f.write('\n'.join(newlines))

# filename = '../data_object_special/simmc2_dials_dstc10_train_predict.txt'
# targetfilename = '../data_object_special/simmc2_dials_dstc10_train_predict_act.txt'
# newlines = []
# with open(filename, encoding="utf-8") as f:
#     for line in f.read().splitlines():
#         if (len(line) > 0 and not line.isspace()):
#             i = 0
#             while line[i-5:i] != 'te : ':
#                 if line[i] == 'C' and line[i-1] == 'O' and line[i-2] == 'N' and line[i-3] == '<':  # <NOCOREF>
#                     line = line[:i+6] + '<ACT>' + line[i+6:]
#                 i += 1
#             newlines.append(line)

# with open(targetfilename, 'w') as f:
#     f.write('\n'.join(newlines))