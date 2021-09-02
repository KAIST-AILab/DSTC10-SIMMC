# DSTC10 SIMMC Repository

This repository contains refactored codebase for DSTC10 SIMMC 2.0 Challenge.

## Change Log
- [September 2, 2021] Fixed error in generated sequence post-processing which led to abnormally high joint-accuracy. Increased generation length (in some cases, response could not be generated, hence leading to error at `.split` in `parse_response` function). This may lead to longer evaluation time (still under 10 minutes).


## Dataset
### Overview
Dataset can be downloaded with `download.sh`. This script was made for bypassing git-lfs quota. Make sure to install `gdown` by `pip install gdown`. It will download then extract data in the following directory format:

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

**NOTE**: Some of the scene images are corrupted and therefore ignored during preprocessing.
```
./data/images/cloth_store_1416238_woman_4_8.png
./data/images/cloth_store_1416238_woman_19_0.png
./data/images/cloth_store_1416238_woman_20_6.png
```

## Project (Codebase)

### Basic structure
All models can be trained and evaluated with `run_{model}.py` script. The directory for each model is under its respective directory, e.g. `baseline`, `caption`, etc. Each model directory has the following format:

```
|-- run_model.py
|-- model
|   |-- __init__.py
|   |-- data            # Directory containing processed data and special tokens -- generated
|   |-- logs            # Directory containing logs -- generated
|   |-- checkpoints     # Directory containing checkpoints -- generated
|   |-- dataset.py      # Contains LightningDataModule / Dataset class
|   |-- evaluate.py     # Contains helpers for evaluation
|   |-- modules.py      # Contains LightningModule (model) class
|   |-- options.py      # Contains supported parsing arguments
|   `-- process.py      # Contains pre-/post-processing func. called in `dataset.py`
`-- ...
```

### Running
In order to run the model, you can either provide `--do-train` (fitting), `--do-test` (testing), or `--do-tune` (learning rate tuning). The codebase is built on `pytorch-lightning`, which provides a device-agnostic boilerplate for training. For distributed training, you can provide `--gpus` argument to specify the gpus. If `--ddp` argument is not provided, data parallel (DP) will be run. It is generally recommended to run distributed data parallel (DDP) for training.

**NOTE**: Multi-word arguments can be passed as either `_` (underscore) or `-` (hyphen), e.g. `--do-test` is equivalent to `--do_test`. 

- Run training
```shell
python run_model.py \
  --do-train \          # Run training
  --gpus="0,1" \        # GPUs on which the script will be run (must be provided as str)
  --ddp \               # Distributed data parallel flag
  --fp16                # AMP -- optional, usually recommended for speed
```

- Run testing (evaluation)
```shell
python run_model.py \
  --do-test \
  --fp16                # One can try DDP, but sampling may become tricky
```

### Extracting Bounding Boxes
For this particular model, we decide not to use object detection models. Instead, we take provided bounding boxes from metadata and create a custom item vector. Items in the `fashion` domain `furniture` domain. Cropped bounding boxes can be extracted from scene images with `image2meta/preprocess.py`. This creates `data/crops` directory, which will be read in using `torchvision.datasets.DatasetFolder` descendent:
```shell
# @1... : fashion @2... : furniture
|-- crops
|   |-- @1210
|   |   |-- 7745184a-a811-4542-98f7-a60c886290d7.png
|   |   |-- 79799209-5102-4fa2-a2e5-c72a378cb0b4.png
|   |   `--...
|   |-- @1211
|   |   |-- 8c459f37-14e5-4c86-af8f-ad4ed8196cf2.png
|   |   |-- 8d255f67-0914-4086-8fff-dcd1427b7e56.png
|   |   `--...
|   `-- ...
`-- ...
```

It also dumps a `token2meta.json` file with the following format.
```
{
  "@1000": {
        "assetType": "blouse_hanging",
        "customerReview": 3.9,
        "availableSizes": [
            "XS",
            "S",
            "XL"
        ],
        "color": "red, white, yellow",
        "pattern": "plaid",
        "brand": "The Vegan Baker",
        "sleeveLength": "long",
        "type": "blouse",
        "price": 39.99,
        "size": "XS",
        "prefab": "1498649_store/Prefabs/_itog039"
    },
    ...
}
```




<!-- ## Disambiguate (Base Boilerplate)
This directory provides a legacy boilerplate for trainer. `Trainer` class under `disambiguate/trainer.py` implements a basic training loop for disambiguation classification task with FP16, distributed training, and Tensoboard/logger monitoring. Training and evaluation loops will be implemented later for end-to-end model as well. 

## End-to-end Multimodal BART (MMBART)
Sequence-to-sequence method for multimodal end-to-end model. This will make use of: 
* Image backbone (ResNet / EfficientNet / ViT family)
* Pretrained Encoder-Decoder Language Model (BART)

The code-base and idea will be largely adopted from MDETR, current SotA for visual question answering among other multimodal tasks. We remove the need for another transformer encoder-decoder on top of BERT-like text encoder by streamlining the whole process with robustly pretrained encoder-decoder model, BART. 

## Simple CLIP (Experimental)
We modify OpenAI's CLIP (Contrastive Language-Image Pretraining) for a simple application. Our main bottleneck in implementing an end-to-end model stems from integrating non-visual metadata; explicitly converting metadata to tokens would lead to excessive memory consumption. Hence, we decide pretrain a joint embedding space between object-specific special tokens and their non-visual metadata. 

The learned joint embeddings can later replace actual token embedding for object special tokens or can be added to token embeddings to provide additional information signal to the model.  -->

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

@article{kamath2021mdetr,
  title={MDETR--Modulated Detection for End-to-End Multi-Modal Understanding},
  author={Kamath, Aishwarya and Singh, Mannat and LeCun, Yann and Misra, Ishan and Synnaeve, Gabriel and Carion, Nicolas},
  journal={arXiv preprint arXiv:2104.12763},
  year={2021}
}

@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```