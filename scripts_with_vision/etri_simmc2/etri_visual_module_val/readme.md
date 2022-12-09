=== SIMMC 2.0 visual module ===
## last updated on 2021.09.10 by Yoonhyung Kim at ETRI

# training configs
* ResNet-34 backbone for feature extraction (ImageNet-1000 pre-trained).
* FC layers for each attribute classifier.
* Thus, the overall pipeline is as follows: (image) => [feature extractor] => [classifier]x5 => (attribute prediction)x5
* Number of categories: assetType 11, color 63, pattern 36, sleeveLength 6, type 18 (see txt/category_list folder).
* bbox image sample whose width or height is less than 20 pixels was excluded from train/val sets.

# validation set configs
* (split1) part2/cloth_store_3_15_0~3-20-11 (scene 53 imgs)
* (split2) part1/cloth_store_paul_19_0~20-10 (scene 21 imgs)
* (split3) part1/cloth_store_woman_17_0~20-10 (scene 40 imgs)
* The rest of the images were used for training.

# evaluation codes
* eval_val_splits.py => evaluate the three validation splits
* eval_single_image.py => evaluate a single bbox image
