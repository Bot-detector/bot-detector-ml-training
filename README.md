# -bot-detector-ML-training
The goal is to provide a training framework for trainign & discovering new models
# setup
creating a python venv to work in and install the project requirements




```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
if pyarrow fails on 3.11
```
pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ --prefer-binary --pre pyarrow
```