import numpy as np
from model import crnn
from data.dataset import get_train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

import sys

model = crnn()

tensorboard = TensorBoard(log_dir="logs/fold_{}".format(sys.argv[1]))

train_input, train_label, test_input, test_label = get_train_test_split(test_fold = list(map(int,sys.argv[1].split(','))), using_CRNN = True)


model.fit(
    [train_input],
    [train_label],
    validation_data=[[test_input], [test_label]],
    epochs=100,
    batch_size=200,
    callbacks=[tensorboard]
    )
