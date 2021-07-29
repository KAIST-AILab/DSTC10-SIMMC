import os
from pathlib import Path


def find_data_dir(root_dir_name=""):
    """
        For finding the right data folder, since directories of data folder are different among users
    """

    assert root_dir_name, "you must give root_dir_name"
    now_dir = Path(os.getcwd())
    for i in range(7):  # arbitrary number. root folder should be in 7 near depth from current directory.
        now_last_dir = str(now_dir).split('/')[-1]
        if now_last_dir == root_dir_name or now_last_dir in root_dir_name:
            break
        now_dir = now_dir.parent.absolute()
    root_dir = str(now_dir)  # now you found full path of root_dir_name

    for path, dirs, files in os.walk(root_dir):  # DFS way walk
        if 'data' in dirs:  # consider as data folder if it has these 3 files
            if (os.path.isfile(os.path.join(path, 'data', 'simmc2_dials_dstc10_train.json')) \
            and os.path.isfile(os.path.join(path, 'data', 'simmc2_dials_dstc10_dev.json')) \
            and os.path.isfile(os.path.join(path, 'data', 'simmc2_dials_dstc10_devtest.json'))  
            ):
                return os.path.join(path, 'data')
    return 






