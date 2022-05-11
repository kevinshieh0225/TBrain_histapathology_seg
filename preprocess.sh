sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

python ./preprocess/preprocess.py
python ./preprocess/prepare_crop.py
python ./preprocess/distribution.py