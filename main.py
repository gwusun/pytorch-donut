import os

import numpy as np
import pandas as pd
import torch
import random

from sklearn.model_selection import train_test_split

from DonutPytorch import Donut, proprocess, print_metrics, plot_predict
import numpy

from utils.data_utils import load_cv_data,get_sliding_windows
from utils.util_pytorch import disable_pytorch_random
disable_pytorch_random()
config={
    "window_size":10,
    "epoch":1
}

# Divide a dataset into training,  validation,  testing sets, whose ratios are 60%,20%,20% respectively.
# For five fold
for f_train_x, f_train_y, f_train_missing, f_test_x, f_test_y, f_test_missing, mean, std in load_cv_data("datasets/cpu4.csv"):
    X_train, X_valid, y_train, y_valid, mssing_train, missing_valid= train_test_split(f_train_x, f_train_y, f_train_missing, test_size=0.2, shuffle=False)
    win_train_x=get_sliding_windows(X_train, window_size=config["window_size"], label=y_train, missing=mssing_train, include_anomaly=False)
    win_train_y=get_sliding_windows(y_train, window_size=config["window_size"], label=y_train, missing=mssing_train, include_anomaly=False)

    win_valid_x=get_sliding_windows(X_valid, window_size=config["window_size"], label=y_valid, missing=missing_valid, include_anomaly=True)
    win_valid_y=get_sliding_windows(y_valid, window_size=config["window_size"], label=y_valid, missing=missing_valid, include_anomaly=True)


    win_test_x=get_sliding_windows(f_test_x, window_size=config["window_size"], label=f_test_y, missing=f_test_missing, include_anomaly=True)
    win_test_y=get_sliding_windows(f_test_y, window_size=config["window_size"], label=f_test_y, missing=f_test_missing, include_anomaly=True)


    model = Donut(window_size=config["window_size"],
                  n_epoch=config['epoch'],
                  number_of_neural_per_layer=10)

    # Train model
    model.fit(win_train_x, win_train_y, valid_x=win_valid_x, valid_y=win_valid_y)

    # Predict model
    scores,modified_score, x_bar_mean, x_bar_std=model.predict(win_test_x, win_test_y)
    # Metrics
    print_metrics(np.array(win_test_y[:, -1]), modified_score)

    # Show the predict
    plot_predict(win_test_x[:, -1], win_test_y[:, -1], x_bar_mean, x_bar_std, modified_score)
