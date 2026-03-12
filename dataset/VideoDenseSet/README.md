# UVO Video Dense Set #

## Annotation format
UVO's instance mask annotations follow [COCO](https://cocodataset.org/#home) and [YTVIS](https://youtube-vos.org/dataset/vis/) annotation format. There are four attributes in ```.json``` files: ``` info ```, ``` annotations ```, ``` videos ```, ```categories```.

### Information for videos
```uvo_json["videos"]``` is a list of dictionaries, each containing information for 1 video clip. The format follows:

```
"start_idx": 210, # at which frame does the clip start
"id": 1, # video id for this video in this json annotation. id is not unique across sparse and dense annotations.
"height": 480, # height of the video
"width": 656, # width of the video
"ytid": "-34DlWSfs2g", # the youtube video id where this video comes from
"label": 390, # original kinetics label for this video
"file_names": [
    "-34DlWSfs2g/210.png",
    ...
    "-34DlWSfs2g/299.png"
] # file names follow the format of {ytid}{frame_idx}
```

If you stored ``` Kinetics400 ``` differently, we recommend you to modify ```file_names``` or build a mapper to your own storage of the dataset.

### Information for annotated masks
```uvo_json["annotations"]``` is a list of dictionaries, each containing information for 1 instance annotation across a video clip. The format follows:

```
"category_id": 1, # Playset does not provide instance label, so every object has the same category 1
"height": 480,
"width": 656,
"video_id": 1, # which video this instance is annotated on
"id": 1, # instance id for this object; not unique across dense and sparse
"areas": [61412, ..., 75707], # sizes of instance size per frame
"bboxes": [
    [0.0, 63.0, 579.0, 416.0],
    ...,
    [0.0, 0.0, 656.0, 480.0],
], # bounding boxes of the instance per frame
"segmentations": [
    {
        "size": [480, 656],
        "counts": "o1e2...AhjS1"
    },
    ...
], # run-length encoding (RLE) of the instance mask per frame
```

## Loading the annotation and evaluation
We recommend following the [tools in YTVIS](https://github.com/youtubevos/cocoapi) for loading the json annotations and run evaluations. To run evaluation for a class-aware model (eg. predict 40 YouTube-VIS categories) in a class-agnostic fashion, we suggest to use ``` useCats=0 ```.

## Baselines
We train a class-agnostic [MaskTrack R-CNN](https://github.com/youtubevos/MaskTrackRCNN).

We used COCO17 Instance Segmentation pretrained backbone to initialize. Finetuning on UVO dense training video provides **15.1** AverageRecall@100 and **9.8** AveragePrecision@100. The models is trained for 12 epochs with batch size 64 and an initial learning rate 0.005, decayed by 10x after epoch 8 and 11.

On test set, the model achieves **11.8** AverageRecall@100 and **7.3** AveragePrecision@100.
