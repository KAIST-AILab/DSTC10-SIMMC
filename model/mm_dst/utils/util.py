import os
import json
import cv2
from pathlib import Path


def find_data_dir(root_dir_name=""):
    """
        For finding the right data folder, since directories of data folder are different among users
        root_dir_name: root directory name of simmc2
        here, data_dir means the original data folder, not folder of preprocessed data
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


def given_bbox_crop_image(image_folder:str, output_folder:str, json_folder:str = '', json_file:str = ''):
    """
    Must give folder name or file name argument in absolute path
    image_folder: folder that all the images are in
    output_folder: folder that cropped output images will be in. the structure is as below
    output_folder -- scene_name1_folder - idividual cropped image files
                  |_ scene_name2_folder - idividual cropped image files
                  ...
    json_folder: if json_folder is given, this will process cropping for all the jsonfiles in that folder,
    json_file: if only json_file is given, this will process croppping only for that particular json file scene (as json filename is scene name)

    Sometimes there are errors at bbox coordinates, scene.json is missing, or image is blank or not openable -> therefore rarely some crops won't be generated
    """
    assert json_folder or json_file, 'Folder_name or image_name should be given!'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def scene_name_given_jsonfile(json_filename):
        return json_filename.rsplit('_', 1)[0] if json_filename.endswith('json') or json_filename.endswith('scene') else json_filename

    def crop(scene_name='', json_file=''):
        if scene_name:
            if not os.path.exists(os.path.join(output_folder, f'{scene_name}')):
                os.makedirs(os.path.join(output_folder, f'{scene_name}'))
            try:
                with open(f'{json_folder}/{scene_name}_scene.json', 'r') as f_in:
                    objs = json.load(f_in)['scenes'][0]['objects']
            except FileNotFoundError:
                print(f'No file named {json_folder}/{scene_name}_scene.json!!')
                return
            image_filename = f'{scene_name[2:]}.png' if scene_name.startswith('m_') else f'{scene_name}.png'
        elif json_file:
            try:
                with open(json_file, 'r') as f_in:
                    objs = json.load(f_in)['scenes'][0]['objects']
            except FileNotFoundError:
                print(f'No file named {json_file}!!')
                return
            scene_name_from_jsonfile = json_file.rsplit('_', 1)[0].rsplit('/', 1)[1]
            if not os.path.exists(os.path.join(output_folder, f'{scene_name_from_jsonfile}')):
                os.makedirs(os.path.join(output_folder, f'{scene_name_from_jsonfile}'))
            image_filename = f'{scene_name_from_jsonfile[2:]}.png' if scene_name_from_jsonfile.startswith(
                'm_') else f'{scene_name_from_jsonfile}.png'

        image = cv2.imread(os.path.join(image_folder, image_filename))
        # some images are not read or blank so causing "libpng error: IDAT: CRC error"
        if image is not None:
            for obj in objs:
                print(obj['index'])
                index = obj['index']
                x_0, y_0, h, w = obj['bbox']
                print([y_0,y_0+h, x_0,x_0+w])
                if h>0 and w>0:
                    crop = image[y_0:y_0+h, x_0:x_0+w]
                    if scene_name:
                        cv2.imwrite('{}.png'.format(os.path.join(output_folder, scene_name, str(index))), crop)
                    elif json_file:
                        cv2.imwrite('{}.png'.format(os.path.join(
                            output_folder, scene_name_from_jsonfile, str(index))), crop)
                else:
                    print('mislabled bbox!')

    if json_folder:
        for filename in os.listdir(json_folder):
            if filename.endswith('.json'):
                scene_name = scene_name_given_jsonfile(filename)
                print('scene_name', scene_name)
                crop(scene_name=scene_name)
    else:
        crop(json_file = json_file)


if __name__ == '__main__':
    image_folder = '/home/haeju/Dev/dstc/dstc10/DSTC10-SIMMC/data/images'
    output_folder = '/home/haeju/Dev/dstc/dstc10/DSTC10-SIMMC/data/cropped_output'
    # json_file = '/Users/HeyJude/Development/GSAI/dstc/dstc10/DSTC10-SIMMC/data/jsons/m_cloth_store_1416238_woman_14_0_scene.json'
    json_folder = '/home/haeju/Dev/dstc/dstc10/DSTC10-SIMMC/data/jsons'
    # given_bbox_crop_image(image_folder=image_folder, output_folder=output_folder, json_file=json_file)
    given_bbox_crop_image(image_folder=image_folder, output_folder=output_folder, json_folder=json_folder)




