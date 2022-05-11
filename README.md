# TBrain_histapathology_seg
TBrain histapathology segmentation contest.

## pre-setting
```sh
sh preprocess.sh
```

## ToDo
1. CUDA error: batchsize = 8, imagesize 640 * 1280, backbone can only use resnet50 (efficeintnet cant use)
2. Inference (pytorch lighting): speculate and analyze result
3. Augmentation

## Exp
1. backbone
2. activation function
3. optimizer
4. loss (regularization)
5. Augmentation
6. post processing
