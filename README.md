# KAIST-AIPRLab Submission

## Dataset
### Overview
Download the dataset from [repository][simmc2] via git-lfs. Run the script `rearrange.sh` to rearrange the `data` folder in the following format.

```
|-- images                                          # scene images
|   |-- cloth_store_1_1_1.png
|   |-- cloth_store_1_1_2.png
|   `-- ...
|-- jsons                                           # bbox and scene jsons
|   |-- cloth_store_1_1_1_bbox.json
|   |-- cloth_store_1_1_1_scene.json
|   `-- ...
|-- fashion_prefab_metadata_all.json                # metadata (fashion)
|-- furniture_prefab_metadata_all.json              # metadata (furniture)
|-- simmc2_dials_dstc10_dev.json                    # dialogue data (dev)
|-- simmc2_dials_dstc10_devtest.json                # dialogue data (devtest)
`-- simmc2_dials_dstc10_train.json                  # dialogue data (train)
```

**NOTE**: Some of the scene images are corrupted and therefore ignored. We do not make use of images in this model other than getting image size.
```
./data/images/cloth_store_1416238_woman_4_8.png
./data/images/cloth_store_1416238_woman_19_0.png
./data/images/cloth_store_1416238_woman_20_6.png
```

## Model Parameters
Download the model parameters from drive link below:

* Google Drive [link](https://drive.google.com/drive/folders/1Qup6UCpt-U1v-Q7O8cYZQZddc9mo2d3U)

## **Subtask 1** : Disambiguation Classification
1. Move into `model/disambiguate`. Running the model automatically preprocesses data.

```shell
python run.py \
  --do-train # training
  --do-test  # evaluating
  --checkpoint / --pretrained_checkpoint # checkpoint path (default: roberta-large)
```

## **Subtask 2, 3, 4** : Multimodal Coreference Resolution, Dialogue State Tracking, Repsonse Generation
Our model is an end-to-end generative model based on BART. Move into `model/mm_dst/bart_dst/scripts`.

1. Preprocess data with
```shell
bash make_data_object_special.sh
```

2. Run training
```shell
bash run_bart_objvec_nocoref.sh
```

3. Run generation / evaluation.
```shell
bash run_bartobjvec_nocoref_gen.sh
```

## **Subtask 5** : Response Retrieval


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