# TWCC update opencv related package
# sudo apt-get update
# sudo apt-get install ffmpeg libsm6 libxext6  -y

# Generate mask and training/validation list
# python ./makemask.py
# python ./splitlist.py

# Use pipenv, or download the package describe in file with pip
pipenv install -r ./requirments/Pipfile
