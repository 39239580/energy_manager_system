import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class DatasetSplitUtils(object):  # 顺序切割数据集
    def __init__(self, train_ratio=0.7, test_ratio=None, split_type='sequential', shuffle=True, random_state=None):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.split_ratio = 1 - train_ratio if test_ratio is None else test_ratio
        self.split_type = split_type
        self.shuffle = shuffle
        self.random_state = random_state

    def split_dataset_sequential(self, x, y=None, index=None):
        if index is None:
            index = int(len(x) * (1 - self.split_ratio))
        train_data = x[:index]
        test_data = x[index:]
        if y is not None:
            train_label = y[:index]
            test_label = y[index:]
            return train_data, test_data, train_label, test_label
        else:
            return train_data, test_data

    def split_dataset_random(self, x, y=None):
        if y is None:
            return train_test_split(x, test_size=self.split_ratio, random_state=self.random_state, shuffle=self.shuffle)
        else:
            return train_test_split(x, y, test_size=self.split_ratio, random_state=self.random_state, shuffle=self.shuffle)

    def split_dataset(self, x, y=None, index=None):
        if self.split_type == 'sequential':
            return self.split_dataset_sequential(x, y, index=index)
        elif self.split_type == 'random':
            return self.split_dataset_random(x, y)
        else:
            raise ValueError("Invalid split type: {}".format(self.split_type))


if __name__ == '__main__':
    data_x = np.random.random((100, 4))
    data_y = np.random.random((100,))
    print("测试1：顺序切割数据集")
    split_utils = DatasetSplitUtils(test_ratio=0.3, split_type='sequential', random_state=42)
    X_train, X_test, y_train, y_test = split_utils.split_dataset(data_x, data_y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train, y_train)
    print(X_test, y_test)
    print("测试2：随机切割数据集")
    split_utils = DatasetSplitUtils(test_ratio=0.3, split_type='random', random_state=42)
    X_train, X_test, y_train, y_test = split_utils.split_dataset(data_x, data_y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(X_train, y_train)
    print(X_test, y_test)