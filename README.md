## Overview
This project is fork from [ultralytics/yolov5](https://github.com/ultralytics/yolov5) and training on the [**Street View House Numbers (SVHN)**](http://ufldl.stanford.edu/housenumbers/) Dataset. I only use the Original images with character level bounding boxes.
![](https://i.imgur.com/2HzOXtb.jpg =x300).

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1) 

## Installation
All requirements should be detailed in requirements.txt.
```
# python version: Python 3.6.9
pip3 install -r requirements.txt
```
To get more information on environvent setup, refer to [Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## Dataset Preparation
1. download the training data from [**Street View House Numbers (SVHN)**](http://ufldl.stanford.edu/housenumbers/) 
2. `SVHN2YOLO5.py` can help you convert the meta file of SVHN to the labels' format required.
```bash
$ python3 SVHN2YOLO5.py -h
    "usage: SVHN2YOLO5.py [-h] [--svhn_meta_file IN_FILE]
                         [--yolov5_meta_dir OUT_DIR]

    optional arguments:
      -h, --help                 show this help message and exit
      --svhn_meta_file IN_FILE   path to digitStruct.mat
      --yolov5_meta_dir OUT_DIR  path to yolov5 labels' dir"
```
3. [Organize Directories](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#3-organize-directories)

## Training
- Using the following script to get more information
```
$ python train.py --help
```
- Example
```
python3 train.py --img 480 --batch 50 --epochs 50 --data SVHN.yaml --weights yolov5m.pt --device 0 --workers 10
```
- Here I use the default hyperparameter.
    - In the begining, I use the config file `data/hyp.scratch.yaml` till model converge.
    - And then, use `data/hyp.finetune.yaml` to finetune the model

- To visualize the train process
    `tensorboard --logdir <path_to_project_name># default project name is runs/train/exp`
    ![](http://)
    
## Testing
- Using the following script to get more information
```
$ python detect.py --help
```
- example
```bash
python3 detect.py --weight "<path_to_model_weight.pt>" --source "<path_to_img or path_to_img_dir>" --device 0 --save-txt --save-conf
```
- The result would save under `runs/detect/exp`

## Reference
This project is fork from [ultralytics/yolov5](https://github.com/ultralytics/yolov5).