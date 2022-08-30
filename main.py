import numpy as np
import pandas as pd
import torch
import random
from DonutPytorch import Donut, proprocess, print_metrics, plot_predict
import numpy

# for reproduce
torch.random.manual_seed(3)
random.seed(3)
numpy.random.seed(3)

config={
    "window_size":70,
    "epoch":240
}

# Read data from file
kpi = pd.read_csv("datasets/cpu4.csv")
x, y = kpi['value'].values, kpi['label'].values
assert len(x) == len(y)



# Divide a dataset into training,  validation,  testing sets, whose ratios are 60%,20%,20% respectively.
train_spilt, val_spilt = 0.6, 0.8
train_x, valid_x, test_x = x[: int(len(x) * train_spilt)], x[int(len(x) * train_spilt): int(len(x) * val_spilt)], x[int(len(x) * val_spilt):]
train_y, valid_y, test_y = y[: int(len(y) * train_spilt)], y[int(len(y) * train_spilt): int(len(y) * val_spilt)], y[int(len(y) * val_spilt):]
train_x, train_y = proprocess(train_x, train_y,slide_win=config["window_size"])
valid_x, valid_y = proprocess(valid_x, valid_y,slide_win=config["window_size"])
test_x, test_y = proprocess(test_x, test_y,slide_win=config["window_size"])


model = Donut(window_size=config["window_size"],n_epoch=config['epoch'])

# Train model
model.fit(train_x, train_y, valid_x=train_x, valid_y=train_y)

# Predict model
scores,modified_score, x_bar_mean, x_bar_std=model.predict(test_x, test_y)

# Metrics
print_metrics(np.array(test_y[:, -1]), modified_score)



# Show the predict
plot_predict(test_x[:, -1], test_y[:, -1], x_bar_mean, x_bar_std, modified_score)
