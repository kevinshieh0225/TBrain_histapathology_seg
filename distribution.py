import os, cv2
from tqdm import tqdm
import numpy as np
from config import load_setting
from sklearn.cluster import KMeans

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
    root = load_setting()['root']
    img_folder = os.path.join(root, 'SEG_Train_Datasets', 'Train_Images')
    respath = './cfg/distrubution.csv'
    img_id_list, rbg_list = rgbw_map(img_folder)

    rgbdstb = rbgcluster(8, img_id_list, rbg_list)
    rgbdstb = dict(sorted(rgbdstb.items(), key=lambda item: item[1]))



    # RGB normalization
    # mean: [210.92679284 158.3909311  196.35660441]
    # std: [43.74901446 77.66571087 49.26829445]

    # mean: [0.82716389 0.62114091 0.7700259 ]
    # std: [0.17156476 0.30457142 0.193209  ]