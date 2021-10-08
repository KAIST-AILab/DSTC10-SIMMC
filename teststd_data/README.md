# Teststd dataset 

## Dataset
### Overview
Download the dataset from [repository][simmc2] via git-lfs. Put all scene images from `simmc2_scene_images_dstc10_teststd.zip` to the directory `../data/images`. Put all scene jsons from `simmc2_scene_jsons_dstc10_teststd.zip` to the directory `../data/jsons`

```
|-- images                                                              # scene images
|   |-- cloth_store_1_1_1.png
|   |-- cloth_store_1_1_2.png
|   `-- ...
|-- jsons                                                               # bbox and scene jsons
|   |-- cloth_store_1_1_1_bbox.json
|   |-- cloth_store_1_1_1_scene.json
|   `-- ...
|-- fashion_prefab_metadata_all.json                                    # metadata (fashion)
|-- furniture_prefab_metadata_all.json                                  # metadata (furniture)
|-- simmc2_dials_dstc10_dev.json                                        # dialogue data (dev)
|-- simmc2_dials_dstc10_teststd_public.json                             # dialogue data (teststd)
|-- simmc2_dials_dstc10_devtest.json                                    # dialogue data (devtest)
|-- simmc2_dials_dstc10_train.json                                      # dialogue data (train)
|-- simmc2_dials_dstc10_teststd_retrieval_candidates_public.json        # retrieval data (teststd)
|-- simmc2_dials_dstc10_dev_retrieval_candidate.json                    # retrieval data (dev)
`-- simmc2_dials_dstc10_devtest_retrieval_candidate.json                # retrieval data (devtest)
```

**NOTE**: Some of the scene images are corrupted and therefore ignored. We do not make use of images in this model other than getting image size.
```
./data/images/cloth_store_1416238_woman_4_8.png
./data/images/cloth_store_1416238_woman_19_0.png
./data/images/cloth_store_1416238_woman_20_6.png
```

## **Data Preprocessing**
For our model input, preprocess the datasets to reformat the data. 

Make sure to download simmc2-data into `./data` before launching the code .
1. Move into `scripts`, run the following command.
For teststd dataset without target(label) file,
```shell
python convert.py \
  --input_path_json=../data/simmc2_dials_dstc10_teststd_public.json \
  --output_path_predict=../teststd_data/teststd_predict.txt \
  --object_special_token_item2id=item2id.json \
  --scene_json_folder=../data/jsons  \
  --image_folder=../data/images \
  --with_target=0
```

## **Evaluation**
All tasks can be evaluated with the same model parameters.

### **(Subtask 1) Disambiguation Classification**

Move to the directory `scripts`.

```shell
python run_bart_multi_task_disambiguation.py \
  --path_output=../teststd_results/dstc10-simmc-teststd-pred-subtask-1.json \
  --prompts_from_file=../teststd_data/teststd_predict.txt \
  --disambiguation_file=../teststd_data/teststd_disambiguation_for_inference.json \
  --item2id item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS> 
```

Disambiguation file, `teststd_disambiguation_for_inference.json` contains just the information about dialogue index and the turn number.

### **(Subtask 2 & 3 & 4-a) MM Coreference Resolution & MM-DST & Response Generation Task** 

Move to the directory `scripts`.

```shell
python run_bart_multi_task_mm_dst.py \
  --prompts_from_file=../teststd_data/teststd_predict.txt \
  --path_output=../teststd_results/dstc10-simmc-teststd-pred-subtask-3.txt \
  --item2id=item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS>
```
 
This script makes the line-by-line *.txt result. To make the generation-task result file from `dstc10-simmc-teststd-pred-subtask-3.txt`, use the following command in the directory `preprocessing_data`. 

 ```shell
python convert_mm_dst_to_response.py \
  --input_path_text=../teststd_results/dstc10-simmc-teststd-pred-subtask-3.txt \
  --dialog_meta_data=../teststd_data/teststd_disambiguation_for_inference.json \
  --output_path_json=../teststd_results/dstc10-simmc-teststd-pred-subtask-4-generation.json
```

### **(Subtask 4-b) Response Retrieval**

Move to the directory `scripts`.

```shell
python run_bart_multi_task_retrieval.py \
  --path_output=../teststd_results/dstc10-simmc-teststd-pred-subtask-4-retrieval.json \
  --prompts_from_file=../teststd_data/teststd_retrieval.txt \
  --candidate_file=../data_object_special/teststd_retrieval.json \
  --item2id item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS>
```

`teststd_retrieval.txt` contains the turns extracted from `teststd` to be evaluated for the retrieval-task.
Candidate file, `teststd_retrieval.json` contains the reformatted candidates, dialogue index and the turn number.

[simmc2]:https://github.com/facebookresearch/simmc2
