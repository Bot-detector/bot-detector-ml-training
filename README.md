# bot-detector-ML-training
The goal is to provide a training framework for trainign & discovering new models

# setup
we will create a python virtual environment with ```python -m venv .venv```
creating a python venv to work in and install the project requirements

## requirements
make sure to have the following on your device:
- python version > 3.10, 
- pip package manager
- git

we suggest you use:
- visual studio code
- github desktop

## installation
we will clone the repository then install the virtual environment

### github desktop
click the green button top right here https://github.com/Bot-detector/bot-detector-ml-training

### command line
clone the repository
```sh
git clone https://github.com/Bot-detector/bot-detector-ml-training.git
```
go into the repository
```sh
cd bot-detector-ml-training
```

### setting up the virtual environment
run the following
```sh
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
if pyarrow fails on 3.11
```sh
pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ --prefer-binary --pre pyarrow
```
