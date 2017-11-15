# CZ4032 Data Analytics and Mining

We participated in [Caterpillar Tube Pricing](https://www.kaggle.com/c/caterpillar-tube-pricing), a Kaggle competition.

## Project Setup

An Unix-based system (Mac/ Linux) is recommended. Tested on Ubuntu 16.04 LTS with Python 2.7.12.

1. Install [XGBoost](http://xgboost.readthedocs.io/en/latest/build.html)
1. Install Python dependencies: `$ pip install --user numpy scipy scikit-learn pandas matplotlib bayesian-optimization` (`bayesian-optimization` is not required if you don't intend to tune hyer-parameters)
1. Run the wrapper script: `$ python wrapper.py -tp`

## Command-line interface

```
$ python wrapper.py -h
usage: wrapper.py [-h] [-p] [-t] [-n] [-c] [-b] [-r] [-e]

optional arguments:
  -h, --help           show this help message and exit
  -p, --predict        do predictions
  -t, --train          train the model
  -n, --no-preprocess  skip preprocessing
  -c, --cv             do cross-validation (no model output)
  -b, --bo             hyper-parameter tuning using BO
  -r, --rf             use random-forest
  -e, --ensemble       ensemble xgboost and random-forest
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

Random Forest: 0.254972

XGBoost: 0.209043 (#19)