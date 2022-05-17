# TBrain_histapathology_seg
TBrain histapathology segmentation contest.

## pre-setting
```sh
sh preprocess.sh
```
## cross validation
```
Name                    f1-score    precision   recall
voting-1                0.899770    0.896443    0.903122
U+_nc_moreaug_FTL_fd0	0.892328    0.886696    0.898032
U+_nc_moreaug_FTL_fd1	0.888392    0.890386    0.886406
U+_nc_ef4ap_FTL_5fd2	0.890481    0.899894    0.881263
U+_nc_ef4ap_FTL_5fd3
U+_nc_ef4ap_FTL_5fd4
```

## Exp
1. backbone
2. optimizer
3. loss (regularization)
4. Augmentation
5. post processing
