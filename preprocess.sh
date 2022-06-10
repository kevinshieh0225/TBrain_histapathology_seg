# for twcc only
# sudo apt-get update
# sudo apt-get install ffmpeg libsm6 libxext6  -y

pip install -r requirments.txt
python ./makemask.py
python ./splitlist.py