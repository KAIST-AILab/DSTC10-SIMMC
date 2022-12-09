# Learning to Embed Multi-Modal Contexts forSituated Conversational Agents

## Overview

Anonymized github repository for NAACL 2022 submission


## **Environment**
Install the conda virtual environment by:
```shell
conda env create -f env.yml
```
Download  `nltk`'s `punkt` model (for response generation evaluation) by:
```shell
python -c "import nltk; nltk.download('punkt')"
```
## **Dataset**

Download the dataset from [repository][simmc2] via git-lfs. Run the script `rearrange.sh` to rearrange the `data` folder in the following format.

```
|-- images                                                # scene images
|   |-- cloth_store_1_1_1.png
|   |-- cloth_store_1_1_2.png
|   `-- ...
|-- jsons                                                 # bbox and scene jsons
|   |-- cloth_store_1_1_1_bbox.json
|   |-- cloth_store_1_1_1_scene.json
|   `-- ...
|-- fashion_prefab_metadata_all.json                      # metadata (fashion)
|-- furniture_prefab_metadata_all.json                    # metadata (furniture)
|-- simmc2_dials_dstc10_dev.json                          # dialogue data (dev)
|-- simmc2_dials_dstc10_devtest.json                      # dialogue data (devtest)
|-- simmc2_dials_dstc10_train.json                        # dialogue data (train)
|-- simmc2_dials_dstc10_dev_retrieval_candidate.json      # retrieval data (dev)
`-- simmc2_dials_dstc10_devtest_retrieval_candidate.json  # retrieval data (devtest)
```

**NOTE**: Some of the scene images are corrupted and therefore ignored. We do not make use of images in this model other than getting image size.
```
./data/images/cloth_store_1416238_woman_4_8.png
./data/images/cloth_store_1416238_woman_19_0.png
./data/images/cloth_store_1416238_woman_20_6.png
```

## **Model Parameters**
Since our model is jointly trained on all tasks, we only need a single model for all subtasks. Download the model parameters by one of the following methods:

1.  Download from Google Drive: [checkpoint-22000.zip](https://drive.google.com/file/d/1ffPkx1bcJrYL7nN88FCXDs5HrUc_SJhJ/view?usp=sharing)
2.  Download with `gdown`
```shell
gdown --id 1ffPkx1bcJrYL7nN88FCXDs5HrUc_SJhJ
```

## **Data Preprocessing**
For our model input, preprocess the datasets to reformat the data. 

Make sure to download simmc2-data into `./data` before launching the code .

1. Move into `scripts`, run the following command.
```shell
python convert.py \
  --input_path_json=<YOUR INPUT PATH JSON> \
  --output_path_predict=<YOUR OUTPATH PREDICT> \
  --output_path_target=<YOUR OUTPATH TARGET> \
  --object_special_token_item2id=item2id.json \
  --scene_json_folder=../data/jsons \
  --image_folder ../data/images \
```
For devtest dataset, 

```shell
python convert.py \
  --input_path_json=../data/simmc2_dials_dstc10_devtest.json \
  --output_path_predict=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --output_path_target=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
  --object_special_token_item2id=item2id.json \
  --scene_json_folder=../data/jsons  \
  --image_folder=../data/images 
```
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
Since our model is jointly trained on all subtasks, the additional target files are needed.

e.g `simmc2_dials_dstc10_train_disambiguation_label.txt`, `simmc2_dials_dstc10_train_response.txt` for disambiguation-task and retrieval-task, respectively. These are already uploaded in the directory `data_object_special`.

## **Training**
Our model is jointly trained with losses from each tasks based on BART.

Make sure to download simmc2-data into `./data` before training: https://github.com/facebookresearch/simmc2/tree/main/data
1. Move into `scripts`, Run training.

```shell
bash run_bart_multi_task.sh
```
or 

```shell
python run_bart_multi_task.py \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --item2id=./item2id.json \
  --train_input_file=../data_object_special/simmc2_dials_dstc10_train_predict.txt \
  --train_target_file=../data_object_special/simmc2_dials_dstc10_train_target.txt  \
  --disambiguation_file=../data_object_special/simmc2_dials_dstc10_train_disambiguation_label.txt \
  --response_file=../data_object_special/simmc2_dials_dstc10_train_response.txt \
  --eval_input_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --eval_target_file=../data_object_special/simmc2_dials_dstc10_devtest_target.txt \
  --output_dir=../multi_task/model \
  --train_batch_size=12 \
  --output_eval_file=../multi_task/model/report.txt \
  --num_train_epochs=10  \
  --eval_steps=3000  \
  --warmup_steps=8000 
```
## **Evaluation**
All tasks can be evaluated with the same model parameters. 

**NOTE**: For `teststd` split, input preprocessing instructions and preprocessed dataset can be found under `teststd_data` directory along with `README.md`.

### **(Subtask 1) Disambiguation Classification**
```shell
bash run_bart_multi_task_disambiguation.sh
```
or

```shell
python run_bart_multi_task_disambiguation.py \
  --path_output=../devtest_results/dstc10-simmc-devtest-pred-subtask-1.json \
  --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --disambiguation_file=../data_object_special/simmc2_dials_dstc10_devtest_inference_disambiguation.json \
  --item2id item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS> 
```

Disambiguation file, `simmc2_dials_dstc10_devtest_inference_disambiguation.json` contains information on dialogue and turn index.

### **(Subtask 2 & 3 & 4-a) MM Coreference Resolution & MM-DST & Response Generation** 
```shell
bash run_bart_multi_task_mm_dst.sh
```
or
```shell
python run_bart_multi_task_mm_dst.py \
  --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --path_output=../devtest_results/dstc10-simmc-devtest-pred-subtask-3.txt \
  --item2id=item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS>
```
This script creates a line-by-line *.txt prediction. To parse the line-by-line results `dstc10-simmc-devtest-pred-subtask-3.txt` into *subtask-4-generation.json format, use the following command in the directory `preprocessing_data`. 

```shell
python convert_mm_dst_to_response.py \
  --input_path_text=../devtest_results/dstc10-simmc-devtest-pred-subtask-3.txt \
  --dialog_meta_data=../data_object_special/simmc2_dials_dstc10_devtest_inference_disambiguation.json \
  --output_path_json=../devtest_results/dstc10-simmc-devtest-pred-subtask-4-generation.json
```

### **(Subtask 4-b) Response Retrieval**
```shell
bash run_bart_multi_task_retrieval.sh
```
or
```shell
python run_bart_multi_task_retrieval.py \
  --path_output=../devtest_results/dstc10-simmc-devtest-pred-subtask-4-retrieval.json \
  --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --candidate_file=../data_object_special/simmc2_dials_dstc10_devtest_retrieval.json \
  --item2id item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS>
```
Candidate file, `simmc2_dials_dstc10_devtest_retrieval.json` contains the reformatted candidates, dialogue and turn index.

## References
```
@article{kottur2021simmc,
  title={SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations},
  author={Kottur, Satwik and Moon, Seungwhan and Geramifard, Alborz and Damavandi, Babak},
  journal={arXiv preprint arXiv:2104.08667},
  year={2021}
}

@article{lewis2019bart,
  title={Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension},
  author={Lewis, Mike and Liu, Yinhan and Goyal, Naman and Ghazvininejad, Marjan and Mohamed, Abdelrahman and Levy, Omer and Stoyanov, Ves and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1910.13461},
  year={2019}
}

@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

[simmc2]:https://github.com/facebookresearch/simmc2
