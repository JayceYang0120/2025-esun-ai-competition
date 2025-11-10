#!/bin/bash

python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python preprocess.py
python XGBoost.py