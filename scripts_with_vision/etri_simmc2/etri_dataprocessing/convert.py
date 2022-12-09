import os
import os.path as osp
import json
import pprint
import cv2
import ipdb
fashion_meta_data = json.load(open("/ext/coco_dataset/simmc2/data/fashion_prefab_metadata_all.json"))
furniture_meta_data = json.load(open("/ext/coco_dataset/simmc2/data/furniture_prefab_metadata_all.json"))
dummy_files = []
dummy_images = []
dummy_crop_images = {}
image_folder_list = ["simmc2_scene_images_dstc10_public_part1", "simmc2_scene_images_dstc10_public_part2"]
item2id = json.load(open("/ext/dstc10/yschoi/item2id.json"))
from tqdm import tqdm
items = []
statistics_item_bboxes = {}

scene_images = {}

def save_crop_image(scene_root_dir, image_root_dir, crop_image_root_dir):
    file_list = os.listdir(scene_root_dir)
    for scene_name in tqdm(file_list, "Converting"):
        if "bbox" in scene_name:
            continue
        scene_data = json.load(open(osp.join(scene_root_dir, scene_name)))
        image_file = scene_name[:-11] + ".png" # without _scene.json 

        exist = False
        for folder in image_folder_list:
            image_path = os.path.join(image_root_dir, folder, image_file)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    height, width = image.shape[:2]
                else:
                    dummy_images.append(image_path)
                    print(f"Dummy Images : {image_path}")
                exist = True
                break

        if not exist:
            dummy_files.append(scene_name)
            print(f"Does not exsit Image file : {scene_name}")
            continue

        for idx, obj in enumerate(scene_data["scenes"][0]['objects']):
            prefab_path = obj["prefab_path"]
            obj_idx = obj['index']
            special_token = item2id[prefab_path]
            if item_id not in items:
                os.makedirs(osp.join(crop_image_root_dir, item_id), exist_ok=True)
                statistics_item_bboxes[item_id] = 0
                items.append(item_id)
            x, y, h, w = obj['bbox']
            # unique_id = obj["unique_id"]
            # index = obj["index"]
            left = x
            right = x + w 
            bottom = y 
            top = y + h
            crop = image[bottom:top, left:right]
            tag = 0
            crop_img_name = osp.join(crop_image_root_dir, item_id, scene_name[:-5]+f"_{tag}.png") # {crop_image_root_dir}/{token name}/scene_name
            while osp.exists(crop_img_name):
                tag += 1
                crop_img_name = osp.join(crop_image_root_dir, item_id, scene_name[:-5]+f"_{tag}.png") 
                # print(f"Already exist img, override {crop_img_name}")
            try:
                cv2.imwrite(crop_img_name, crop)
                statistics_item_bboxes[item_id] += 1
            except:
                print(f"Error crop size : {crop_img_name}, shape : {crop.shape}")
                dummy_crop_images[crop_img_name] = {
                    "unique_id" : obj["unique_id"],
                    "index" : obj["index"],
                    "bbox" : obj['bbox'],
                }
    json.dump(statistics_item_bboxes, open("statistics_items_with_m.json", "w"), indent=4)
    json.dump(dummy_crop_images, open("dummy_crop_images.json", "w"), indent=4)
    pprint.pprint(statistics_item_bboxes)
if __name__ == "__main__":
    # Image Test
    ###########################################################################
    from PIL import Image
    import cv2
    img_name = "/ext/coco_dataset/simmc2/data/simmc2_scene_images_dstc10_public_part1/cloth_store_1416238_woman_9_2.png"
    json_data_name = "/ext/coco_dataset/simmc2/data/public/m_cloth_store_1416238_woman_9_2_scene.json"

    json_data = json.load(open(json_data_name))
    img = cv2.imread(img_name)
    print(img.shape)
    for obj in json_data["scenes"][0]['objects']:
        print(obj["bbox"])
        x, y, h, w = obj["bbox"] # x, y top left
        left = x
        right = x + w 
        bottom = y 
        top = y + h
        start_point = (left, bottom)
        end_point = (right, top)
        # print(f"Y : {bottom, top}")
        # print(f"X : {left, right}")
        print(obj["index"])
        color = (255, 255, 255)
        thickness = 2
        cv2.rectangle(img, start_point, end_point, color, 2)

        org = (left+5, bottom+15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 2
        cv2.putText(img, str(obj["index"]), org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Cropped", img)
    cv2.waitKey(0)    

    # os.makedirs("/ext/dstc10/crop_images", exist_ok=True)
    # save_crop_image(scene_root_dir="/ext/coco_dataset/simmc2/data/public",
    #                 image_root_dir="/ext/coco_dataset/simmc2/data",
    #                 crop_image_root_dir="/ext/dstc10/yjlee/crop_images"
    # )

    # id2item = {}
    # for k, v in item2id.items():
    #     id2item[v] = k
    # file_list = os.listdir("/ext/dstc10/crop_images")
    # for file in file_list:
    #     if file =="information":
    #         continue
    #     item_name = id2item[f"<{file}>"]
    #     if "wayfair" in item_name:
    #         annotation = furniture_meta_data[item_name]
        
    #     else : 
    #         try:
    #             annotation = fashion_meta_data[item_name]
    #         except:
    #             ipdb.set_trace()
        
    #     annotation["item_name"] = item_name
        # json.dump(annotation, open(os.path.join("/ext/dstc10/crop_images", file, "annotation.json"), "w"), indent=4)