# CZ4032 Data Analytics and Mining

We participated in [Caterpillar Tube Pricing](https://www.kaggle.com/c/caterpillar-tube-pricing), a Kaggle competition.

## Project Setup

An Unix-based system (Mac/ Linux) is recommended. Tested on Ubuntu 16.04 LTS with Python 2.7.12.

1. Install [XGBoost](http://xgboost.readthedocs.io/en/latest/build.html)
1. Install Python dependencies: `$ pip install --user numpy scipy scikit-learn pandas matplotlib bayesian-optimization` (`bayesian-optimization` is not required if you don't intend to tune hyper-parameters)
1. Run the wrapper script: `$ python wrapper.py -tp xgb`

## Command-line interface

```
$ python wrapper.py -h
usage: wrapper.py [-h] [-n] [-p] [-t] [-c] [-b] {xgb,xgbk,rf,xgbrf}

positional arguments:
  {xgb,xgbk,rf,xgbrf}  method use for train and predict

optional arguments:
  -h, --help           show this help message and exit
  -n, --no-preprocess  skip preprocessing
  -p, --predict        do predictions
  -t, --train          train the model
  -c, --cv             cross-validation (no model output) (only for xgb)
  -b, --bo             hyper-parameter tuning using BO (only for xgb)
```

## Repository Structure

```
competition_data/
    [competition data from Kaggle]
model/
    [model files]
viz/
    [visualizations]
config.json             # hyper-parameters for xgboost
visualize.py            # produce visualizations
preprocess.py           # preprocessing script (can be run stand-alone)
train.py                # training
predict.py              # predictions
wrapper.py              # wrapper
```

## Evaluation Score (Private Leaderboard)

| Method | Command-line Equivalent | RMSLE | Rank
| --- | --- | --- | ----
| Random Forest | `rf` | 0.254972 | #681
| XGBoost | `xgb` | 0.209043 | #19
| Ensemble (XGBoost + RF) | `xgbrf` | 0.209040 | #19
| Ensemble (10-fold XGBoost) | `xgbk` | 0.208923 | #16 