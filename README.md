# TBrain_histapathology_seg
TBrain histapathology segmentation contest.

## pre-setting
```sh
sh preprocess.sh
```
## cross validation
```
Name                        f1-score    precision   recall
voting-1                    0.899770    0.896443    0.903122
U+_nc_moreaug_FTL_fd0	    0.892328    0.886696    0.898032
U+_nc_moreaug_FTL_fd1	    0.888392    0.890386    0.886406
U+_nc_ef4ap_FTL_5fd2	    0.890481    0.899894    0.881263
U+_nc_ef4ap_FTL_5fd3
U+_nc_ef4ap_FTL_5fd4

U+_nc_ef4ap_FTL_10fd0       0.895147    0.883711    0.906883
U+_nc_ef4ap_FTL_10fd2       0.892547    0.885231	0.899985
U+_nc_ef4ap_FTL_10fd3_last  0.894663    0.915445    0.874804
U+_nc_ef4ap_FTL_10fd4       0.899646    0.897439    0.901864
```

## Exp
1. backbone
2. optimizer
3. loss (regularization)
4. Augmentation
5. post processing
