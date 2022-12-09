import os

root = 'txt/category_list'
f_a = open(os.path.join(root, 'assetType.txt'), 'r')
f_c = open(os.path.join(root, 'color.txt'), 'r')
f_p = open(os.path.join(root, 'pattern.txt'), 'r')
f_s = open(os.path.join(root, 'sleeveLength.txt'), 'r')
f_t = open(os.path.join(root, 'type.txt'), 'r')


def match_label(txt_file, label):
    while True:
        line = txt_file.readline()
        if not line: break
        l_name, l_idx = line.rsplit(' ', 1)
        if str(label) == l_idx.rstrip():
            return l_name

def labels_to_category(labels):
    l_a = labels[0]
    l_c = labels[1]
    l_p = labels[2]
    l_s = labels[3]
    l_t = labels[4]

    n_a = match_label(f_a, l_a)
    n_c = match_label(f_c, l_c)
    n_p = match_label(f_p, l_p)
    n_s = match_label(f_s, l_s)
    n_t = match_label(f_t, l_t)

    return [n_a, n_c, n_p, n_s, n_t]






















