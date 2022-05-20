# TBrain_histapathology_seg
TBrain histapathology segmentation contest.

## pre-setting
```sh
sh preprocess.sh
```
pretrain weight is on [GDrive](https://drive.google.com/drive/folders/1UgTa4WhK3WPqX168u9uftHxLsafFYtAZ?fbclid=IwAR0XvHfJDLGW0XNj7SV-Dq4D0_4dPzIKy0RMiGNTokD9Nfc28y0rkT2prD4)

## cross validation
```
Name                        f1-score    precision   recall
voting-1                    0.899770    0.896443    0.903122
U+_nc_moreaug_FTL_fd0	    0.892328    0.886696    0.898032
U+_nc_moreaug_FTL_fd1	    0.888392    0.890386    0.886406
U+_nc_ef4ap_FTL_5fd2	    0.890481    0.899894    0.881263
U+_nc_ef4ap_FTL_5fd3
U+_nc_ef4ap_FTL_5fd4

U+_nc_ef4ap_FTL_10fd0       0.895147    0.883711    0.906883    v
U+_nc_ef4ap_FTL_10fd1       0.886003    0.872314    0.900128
U+_nc_ef4ap_FTL_10fd2       0.892547    0.885231	0.899985
U+_nc_ef4ap_FTL_10fd3_last  0.894663    0.915445    0.874804    v
U+_nc_ef4ap_FTL_10fd4       0.899646    0.897439    0.901864    v
U+_nc_ef4ap_FTL_10fd5       0.894417	0.888879    0.900025
U+_nc_ef4ap_FTL_10fd6       0.898047    0.882558    0.914091    v
U+_nc_ef4ap_FTL_10fd7       0.897821    0.886498    0.909436    v
U+_nc_ef4ap_FTL_10fd8       0.893720    0.891105    0.896352
U+_nc_ef4ap_FTL_10fd9
```

## Exp
1. backbone                 Unetplusplus timm-efficientnetb4 advprop
2. optimizer                SGD with CosineAnnealing Warmup
3. loss (regularization)    Diceloss
4. Augmentation             ColorJitter ElasticTransform Flip ShiftScaleRotate
5. post processing          connectTH
