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
U+_nc_ef4ap_FTL_10fd4_soup5 0.904259    0.899413    0.909157        V

4, 5
base_DL_plus_10fd0          0.89715     0.888939	0.905514
base_DL_plus_10fd3          0.900433    0.906386    0.894558        V
base_DL_plus_10fd7          0.898166	0.889426    0.907079
base_DL_plus_10fd8          0.901063    0.886515    0.916096        V
0, 3, 5, 7,                 X
U+_nc_ef4ap_sDL_10fd4       0.894808    0.868046    0.923273
U+_nc_ef4ap_sDL_10fd6       0.902042    0.917095    0.887474        V
U+_nc_ef4ap_sDL_10fd8       0.902314    0.902158    0.902471        V

U+_nc_ef4ap_FTL_10fd0       0.895147    0.883711    0.906883    4
U+_nc_ef4ap_FTL_10fd1       0.886003    0.872314    0.900128
U+_nc_ef4ap_FTL_10fd2       0.892547    0.885231	0.899985
U+_nc_ef4ap_FTL_10fd3_last  0.894663    0.915445    0.874804    5
U+_nc_ef4ap_FTL_10fd4       0.899646    0.897439    0.901864    1   V
U+_nc_ef4ap_FTL_10fd5       0.894417	0.888879    0.900025    6
U+_nc_ef4ap_FTL_10fd6       0.898047    0.882558    0.914091    2
U+_nc_ef4ap_FTL_10fd7       0.897821    0.886498    0.909436    3
U+_nc_ef4ap_FTL_10fd8       0.893720    0.891105    0.896352    7
U+_nc_ef4ap_FTL_10fd9       0.892483    0.884145    0.900980
```

## Exp
1. backbone                 Unetplusplus timm-efficientnetb4 advprop
2. optimizer                SGD with CosineAnnealing Warmup
3. loss (regularization)    Diceloss
4. Augmentation             ColorJitter ElasticTransform Flip ShiftScaleRotate
5. post processing          connectTH

Good Public Image
Private_00000017
Private_00000018
Private_00000021
Private_00000035
Private_00000039
Private_00000064
Private_00000080
Private_00000129
Private_00000134
Public_00000001
Public_00000003
Public_00000008
Public_00000011
Public_00000012
Public_00000023
Public_00000024
Public_00000025
Public_00000026
Public_00000028
Public_00000034
Public_00000037
Public_00000038
Public_00000041
Public_00000042
Public_00000047
Public_00000061
Public_00000064
Public_00000070
Public_00000080
Public_00000082
Public_00000086
Public_00000088
Public_00000091
Public_00000099
Public_00000102
Public_00000107
Public_00000116
Public_00000120
Public_00000125
Public_00000127