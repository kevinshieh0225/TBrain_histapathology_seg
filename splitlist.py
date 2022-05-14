import os, cv2
from tqdm import tqdm
import csv, json
import numpy as np
import pandas as pd
from utils.config import load_setting
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

NFOLDS = 10
list_json = './trainlist/10fold_list'

def rgbw_map(img_folder):
    img_id_list = [image_id.split('.')[0] for image_id in os.listdir(img_folder)]
    rbg_list = []

    for image_id in tqdm(img_id_list):
        image = cv2.imread(os.path.join(img_folder, f'{image_id}.jpg'))
        rgbimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint32)
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint32)

        rgb_mean = np.mean(rgbimg, axis=tuple(range(rgbimg.ndim-1))) / 255
        white = np.mean(grayimg > 220)
        rgb_mean = np.append(rgb_mean, white)
        rbg_list.append(rgb_mean)

    return img_id_list, rbg_list

def rbgcluster(n_clusters, img_id_list, rbg_list):
    equ = np.array(rbg_list) * 255
    equ = cv2.equalizeHist(equ.astype(np.uint8))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(equ)
    rgbdstb = dict(zip(img_id_list, kmeans.labels_.tolist()))

    return rgbdstb

if __name__ == "__main__":
    ds_cfg = load_setting()
    dataset_root = ds_cfg['dataset_root']
    img_folder = os.path.join(dataset_root, 'Train_Images')
    os.makedirs('./trainlist', exist_ok=True)
    respath = './trainlist/distrubution.csv'
    rgbdstb = {}
    if not os.path.isfile(respath):
        img_id_list, rbg_list = rgbw_map(img_folder)
        rgbdstb = rbgcluster(8, img_id_list, rbg_list)
    else:
        with open(respath, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                rgbdstb[row[0]] = int(row[1])

    id_list = list(rgbdstb.keys())
    cl_list = list(rgbdstb.values())
    skf = StratifiedKFold(n_splits = NFOLDS, random_state = 7, shuffle = True) 
    for idx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(id_list)),cl_list)):
        train_id = pd.DataFrame(id_list).iloc[train_index][0].tolist()
        valid_id = pd.DataFrame(id_list).iloc[val_index][0].tolist()
        train_test_list = {
            'train': train_id,
            'valid': valid_id,
            }
        with open(f'{list_json}_{idx}.json', 'w') as f:  
            json.dump(train_test_list, f)

    # rgbdstb = dict(sorted(rgbdstb.items(), key=lambda item: item[1]))
    
    # group = {}
    # for key, value in rgbdstb:
    #     if key not in group:
    #         group[key] = [value]
    #     else:
    #         group[key].append(value)
    
    # with open(respath, 'w') as f:  
    #     writer = csv.writer(f)
    #     writer.writerow(['id', 'cluster'])
    #     for k, v in rgbdstb.items():
    #         writer.writerow([k, v])

    # with open(jsonpath, 'w') as f:  
    #     json.dumps(group, f)

    # RGB normalization
    # mean: [210.92679284 158.3909311  196.35660441]
    # std: [43.74901446 77.66571087 49.26829445]

    # mean: [0.82716389 0.62114091 0.7700259 ]
    # std: [0.17156476 0.30457142 0.193209  ]