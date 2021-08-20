### Requirement

- transformers

- torch==1.9

- tqdms

- timm

## Dataset

```
# /ext/dstc10/yschoi/train.json
{ "0": # train_index
		{
		"image": image_dir,
 		"predict" :  "{ <@1116>, ... , <@1067> } User : Hi, do you have any jackets today? <EOS>",
 		"belief" : "REQUEST:GET [ type = jacket ] () <  > <EOB>",
 		"objects" : ...,
 		"bbox" : ...,

 		}
```

```
#/ext/dstc10/yschoi/item2id.json
# fashion item <@1000> ~ <@1287>
# furniture itme <@2000> ~ <@2056>
{
    "1498649_store/Prefabs/_itog039": "<@1000>",
    "1498649_store/Prefabs/_itog038": "<@1001>",
    "1208725/Jacket_red": "<@1002>",
    ...
    
}
```

```
#/ext/dstc10/yschoi/simmc2_special_tokens.json
{
    "eos_token": "<EOS>",
    "additional_special_tokens": [
        "<EOB>",
        "<SOM>",
        "<EOM>",
        "materials",
		...
        "brand",
        "color",
        "size",
        "sleeveLength",
        "customerReview",
        "<@1000>",
        "<@1001>",
        "<@1002>",
        "<@1003>",
        "<@1004>",
        }
```



Command with image, text encoder

```
python launch.py --with_image --learning_rate 5e-5
```



Command with only text

```
python launch.py --learning_rate 5e-5
```

