import attr
import json
import glob

import cv2
import numpy as np

from typing import List, Optional
from attr import converters, validators
from os.path import isfile, join

@attr.s
class SceneObject:
    name: str = attr.ib(
        converter=str,
        validator=validators.instance_of(str)
    )
    prefab_path: str = attr.ib(
        converter=str,
        validator=validators.instance_of(str)
    )
    unique_id: int = attr.ib(
        converter=int,
        validator=validators.instance_of(int)
    )
    index: int = attr.ib(
        converter=int,
        validator=validators.instance_of(int)
    )
    bbox: List[int] = attr.ib(
        converter=lambda x: [int(_) for _ in x],
        validator=validators.instance_of(list)
    )
    position: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    up: Optional[List[int]] = attr.ib(  #  should't be dictionary?
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )
    down: Optional[List[int]] = attr.ib(
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )
    left: Optional[List[int]] = attr.ib(
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )
    right: Optional[List[int]] = attr.ib(
        converter=converters.optional(lambda x: [int(_) for _ in x]),
        validator=validators.optional(validators.instance_of(list))
    )

    @bbox.validator
    def check(self, attribute, value):
        if len(value) != 4:
            raise ValueError("bbox must have 4 elements.")
        
    @position.validator
    def check(self, attribute, value):
        if len(value) != 3:
            raise ValueError("position must have 3 elements.")


@attr.s
class CameraObject:
    """
        List because each contains x, y, z position coordinates
    """
    camera: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    right: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    forward: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )
    up: List[float] = attr.ib(
        converter=lambda x: [float(_) for _ in x],
        validator=validators.instance_of(list)
    )


@attr.s
class Scene:
    scene_name: str = attr.ib(
        converter=str,
        validator=validators.instance_of(str)
    )
    scene_image: np.ndarray = attr.ib()
    scene_object: SceneObject = attr.ib()
    camera_object: CameraObject = attr.ib()

    @scene_name.validator
    def check(self, attribute, value):
        bbox_json = join("..", "data", "jsons", "{}_bbox.json".format(value))
        scene_json = join("..", "data", "jsons", "{}_scene.json".format(value))
        scene_image = join("..", "data", "images", "{}.png".format(value))
        if not (
            isfile(bbox_json)\
                and isfile(scene_json)\
                    and isfile(scene_image)
        ):
            raise ValueError("All required scene files do not exist!")

    @classmethod
    def from_json(cls, scene_name: str):
        """
        Input dictionary has the following format 

        *_bbox.json (includes camera):

            {
                "Items": [
                    {
                        "name": <str>,
                        "prefabPath": <str>,
                        "bbox": [<int>, <int>, <int>, <int>],
                        "position": [<float>, <float>, <float>]
                    },
                    ...,
                    {
                        "name": "camera_{}".format(<str>),
                        "prefabPath": "camera",
                        "bbox": [-1, -1, -1, -1],
                        "position": [<float>, <float>, <float>]
                    }
                ]
            }
        
        *_scene.json:

            {
                "scenes": [
                    {
                        "objects": [
                            {
                                "prefab_path": <str>,
                                "unique_id": <int>,
                                "index": <int>,
                                "bbox": [<int>, <int>, <int>, <int>],
                                "position": [<float>, <float>, <float>]
                            },
                            ...
                        ]
                    }
                ],
                "relationships": {
                    "right": {
                        "0": <list<int>>, # Each key is stringified int index
                        ...
                    },
                    "left": ...,
                    "up": ...,
                    "down": ...
                }
            }
        """
        bbox_json = json.load(
            open(
                join("..", "data", "jsons", "{}_bbox.json".format(scene_name))
            )
        )
        scene_json = json.load(
            open(
                join("..", "data", "jsons", "{}_scene.json".format(scene_name))
            )
        )
        image = cv2.imread(join("..", "data", "images", "{}.png".format(scene_name)))

        # Define camera object
        camera_args = dict()
        for item in bbox_json['Items']:
            if item['prefabPath'] == "camera":
                if item['name'] == "camera":
                    camera_args['camera'] = item['position']
                elif item['name'] == "camera_right":
                    camera_args['right'] = item['position']
                elif item['name'] == "camera_forward":
                    camera_args['forward'] = item['position']
                elif item['name'] == "camera_up":
                    camera_args['up'] = item['position']
                else:
                    pass
        camera_obj = CameraObject(**camera_args)

        scene_obj = list()
        # Define scene object
        relation = scene_json['scenes'][0]['relationships']
        for item in scene_json['scenes'][0]['objects']:
            scene_obj_args = dict()
            for k, v in item.items():
                scene_obj_args[k] = v

                if k == "index":
                    # right, left, up, down
                    _right = relation['right'].get(str(v), None)
                    scene_obj_args['right'] = _right
                    _left = relation['left'].get(str(v), None)
                    scene_obj_args['left'] = _left
                    _up = relation['up'].get(str(v), None)
                    scene_obj_args['up'] = _up
                    _down = relation['down'].get(str(v), None)
                    scene_obj_args['down'] = _down

            # Find name from *_bbox.jsons file.
            for bbox_item in bbox_json['Items']:
                if (
                    bbox_item['bbox'] == scene_obj_args['bbox']\
                        and bbox_item['prefabPath'] == scene_obj_args['prefab_path']\
                            and bbox_item['position'] == scene_obj_args['position']
                ):
                    scene_obj_args['name'] = bbox_item['name']            
            scene_obj.append(SceneObject(**scene_obj_args))

        scene_args = {
            'scene_name': scene_name,
            'scene_image': image,
            'scene_object': scene_obj,
            'camera_object': camera_obj
        }
        return cls(**scene_args)
        

@attr.s
class Store:
    scenes: List[Scene] = attr.ib()

    @classmethod
    def from_name(cls, name: str):
        files = glob.glob(
            join(
                "..", "data", "jsons", "{}_*".format(name)
            )
        )
        
        return cls([
            Scene.from_json('_'.join(f.split('/')[-1].split('_')[:-1]).replace(".json", '')) for f in files
        ])

if __name__ == "__main__":
    print(Store.from_name("cloth_store_1_1"))