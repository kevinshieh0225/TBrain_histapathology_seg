# TBrain_histapathology_seg
TBrain histapathology segmentation contest.

## Pre-setting
```sh
$sh preprocess.sh
```
`./trainlist` will fit in the cross validation training/validation set
In current version we put some public/private image into training,
so if you want to re-run training, you have add the specific image into 
training data. 

## Config
Config file for training and inferencing model is in `./cfg`
`setting.yaml` is experiment name, image root, inference path,
                use crossvalidation , use soup setting.
`wandbcfg.yaml` is training model hyperparameter.

## Training
After setting config file, start training:
```sh
$python training.py
```
The training ckpt will save in `./result`
We provide Data Distributed Parallel and will automatically detect gpu number.

## Inference
Modified `inference.py` for to assign inference experiment ckpt.
Use `voting.py` and `embedding.py` to get majority agreement mask.

```sh
$python inference.py
$python voting.py
$python embedding.py
```
Pretrain weight can be download in [GDrive](https://drive.google.com/drive/folders/1UgTa4WhK3WPqX168u9uftHxLsafFYtAZ?fbclid=IwAR0XvHfJDLGW0XNj7SV-Dq4D0_4dPzIKy0RMiGNTokD9Nfc28y0rkT2prD4)

Or best model ckpt in private score is `base_DL_plus_10fd0`

## cross validation result in public
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
