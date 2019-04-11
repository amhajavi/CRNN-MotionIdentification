import numpy as np
from progressbar import progressbar, Percentage, Bar, RotatingMarker, ETA, FileTransferSpeed

widgets_train = [
        'Training data: ', Percentage(),
        ' ', Bar(),
        ' ', ETA()
    ]

widgets_test =  [
        'Testing data: ', Percentage(),
        ' ', Bar(),
        ' ', ETA()
    ]

def split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def split_strip(x):
    return [_.strip() for _ in x.split()]


def zeropad(record , length = 250):
    if len(record) < length:
            pad = np.zeros((length-len(record), 62))
            record = np.concatenate([record, pad])
    return record

def get_10fold():
    with open('data/motion_meta_matrix.txt','r+') as motion_meta:
        motion_data_set = np.array(motion_meta.readlines())
    motion_data_set = list(map(split_strip, motion_data_set))
    return split(motion_data_set, 10)


def get_train_test_split(test_fold = [0,1], using_CRNN = False):
    train_indexes = list(set(range(10)) - set(test_fold))
    folds = np.array(get_10fold())
    train_files_and_labels = np.concatenate(folds[train_indexes])
    test_files_and_labels = np.concatenate(folds[test_fold])
    train_data , test_data, train_label, test_label = [], [], [], []

    for file , label in progressbar(train_files_and_labels, widgets = widgets_train):
        record = np.loadtxt(file)
        try:
            if len(np.shape(record)) < 2:
                continue
            if np.shape(record)[0] < 230:
                continue
            record = zeropad(record, length = 250)
            for index, element in enumerate(record[1:]):
                record[index] = record[index] - record[index-1]
            record[0] = np.zeros(62)
            label = np.eye(144)[int(label)]
        except Exception as e:
            print(file)
            print(np.shape(record))
            raise
        if using_CRNN:
            record = np.reshape(record, (-1, 10, 62))
        else:
            record = np.reshape(record, (-1, 5, 62))
            record = np.mean(record,1, keepdims=True)
            record = np.reshape(record, (-1, 62))
        train_data.append(record)
        train_label.append(label)

    for file , label in  progressbar(test_files_and_labels, widgets = widgets_test):
        record = np.loadtxt(file)
        try:
            if len(np.shape(record)) < 2:
                continue
            if np.shape(record)[0] < 230:
                continue
            record = zeropad(record, length = 250)
            for index, element in enumerate(record[1:]):
                record[index] = record[index] - record[index-1]
            record[0] = np.zeros(62)
            label = np.eye(144)[int(label)]
        except Exception as e:
            print(file)
            print(np.shape(record))
            raise
        if using_CRNN:
            record = np.reshape(record, (-1, 10, 62))
        else:
            record = np.reshape(record, (-1, 5, 62))
            record = np.mean(record,1, keepdims=True)
            record = np.reshape(record, (-1, 62))
        test_data.append(record)
        test_label.append(label)

    return list(train_data), list(train_label), list(test_data), list(test_label)
