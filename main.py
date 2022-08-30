import numpy as np
import pandas as pd
import torch
import random
from DonutPytorch import Donut, proprocess, best_threshold, plot_predict
import numpy

# for reproduce
torch.random.manual_seed(3)
random.seed(3)
numpy.random.seed(3)

# Read data from file
kpi = pd.read_csv("datasets/cpu4.csv")
x, y = kpi['value'].values, kpi['label'].values
assert len(x) == len(y)

# Divide a dataset into training, validation, testing sets, whose ratios are 60%,20%,20% respectively.
train_spilt, val_spilt = 0.6, 0.8
train_x, valid_x, test_x = x[: int(len(x) * train_spilt)], x[int(len(x) * train_spilt): int(len(x) * val_spilt)], x[int(len(x) * val_spilt):]
train_y, valid_y, test_y = y[: int(len(y) * train_spilt)], y[int(len(y) * train_spilt): int(len(y) * val_spilt)], y[int(len(y) * val_spilt):]
train_x, train_y = proprocess(train_x, train_y)
valid_x, valid_y = proprocess(valid_x, valid_y)
test_x, test_y = proprocess(test_x, test_y)


model = Donut()
# Train model
model.fit(train_x, train_y, n_epoch=250, valid_x=train_x, valid_y=train_y)

# Predict model
scores, x_bar_mean, x_bar_std=model.predict(test_x, test_y)

# Find threshold
best_threshold(np.array(test_y[:, -1]), scores.detach().numpy())

# Show the predict
plot_predict(test_x[:, -1], test_y[:, -1], x_bar_mean, x_bar_std, scores)
