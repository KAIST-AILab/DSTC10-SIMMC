# KAIST-AIPRLab Submission

## Dataset
### Overview
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

## Model Parameters
Download the model parameters from drive link below:

* Google Drive [link](https://drive.google.com/drive/u/0/folders/1P_FTLrxp84gVrUI7HG7d-HttnRh8H9xM)

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
For teststd datset without target(label) file,
```shell
python convert.py \
--input_path_json=../data/simmc2_dials_dstc10_teststd_public.json \
--output_path_predict=../teststd_data/teststd_predict.txt \
--object_special_token_item2id=item2id.json \
--scene_json_folder=../data/jsons  \
--image_folder=../data/images \
--with_target=0
```
Since our model is multi-task trained, the additional target files are needed.

e.g `simmc2_dials_dstc10_train_disambiguation_label.txt`, `simmc2_dials_dstc10_train_response.txt` for disambiguation-task and retrieval-task, respectively. These are alreadey uploaded in the directory `data_object_special`.

## **Train Model**
Our model is jointly trained with losses from each tasks based on BART.

Make sure to download simmc2-data into `./data` before training. 
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

**1. Disambiguation Task**
```shell
bash run_bart_multi_task_disambigutaion.sh
```
or

```shell
python run_bart_multi_task_disambiguation.py \
 --path_output=../results/dstc10-simmc-devtest-pred-subtask-1.json \
 --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
 --disambiguation_file=../data_object_special/simmc2_dials_dstc10_devtest_inference_disambiguation.json \
 --item2id item2id.json \
 --add_special_tokens=../data_object_special/simmc_special_tokens.json \
 --model_dir=<YOUR MODEL CHECKPOINTS> 
```

Disambiguation file, `simmc2_dials_dstc10_devtest_inference_disambiguation.json` containes just the information about dialogue index and the turn number.
 
**2. MM_DST & Response Generation Task** 

```shell
bash run_bart_multi_task_mm_dst.sh
```
or
```shell
 python run_bart_multi_task_mm_dst.py \
  --prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
  --path_output=../results/mm_dst_result.txt \
  --item2id=item2id.json \
  --add_special_tokens=../data_object_special/simmc_special_tokens.json \
  --model_dir=<YOUR MODEL CHECKPOINTS>
```
 
 This script makes the line-by-line *.txt result. To make from line-by-line *.txt to *.json, move into `processing_data`.
 
 ```shell
 python convert_line_to_json_for_mm_dst.py \
  --prediction=../results/mm_dst_result.txt \
  --output=../results/dstc10-simmc-devtest-pred-subtask-3.json
```
To make the generation-task result file, use the following command in the same directory. 

 ```shell
 python convert_mm_dst_to_response.py \
  --input_path_test=../results/mm_dst_result.txt \
  --output_path_json=../results/dstc10-simmc-devtest-pred-subtask-4-generation.json
```

**3. Retrieval Task**

```shell
bash run_bart_multi_task_retrieval.sh
```
or
```shell
python run_bart_multi_task_retrieval.py \
--path_output=../results/dstc10-simmc-devtest-pred-subtask-4-retrieval.json \
--prompts_from_file=../data_object_special/simmc2_dials_dstc10_devtest_predict.txt \
--candidate_file=../data_object_special/simmc2_dials_dstc10_devtest_retrieval.json \
--item2id item2id.json \
--add_special_tokens=../data_object_special/simmc_special_tokens.json \
--batch_size=24 \
--model_dir=<YOUR MODEL CHECKPOINTS>
```
Candidate file, `simmc2_dials_dstc10_devtest_retrieval.json` contains the reformatted canndidates, dialogue index and the turn number.

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