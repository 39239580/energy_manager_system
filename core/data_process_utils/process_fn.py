import sklearn.preprocessing as preprocessing
import joblib as jl
import numpy as np
import pandas as pd


def min_max_scaler(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)


def sk_save(scaler, sk_path):
    jl.dump(scaler, sk_path)


def sk_load(sk_path):
    return jl.load(sk_path)


class BaseScaler(object):
    def __init__(self, **kwargs):
        self.scaler = None
        self.kwargs = kwargs

    def fit(self, x, y=None, **fit_params):
        return self.scaler.fit(x, y, **fit_params)

    def fit_transform(self, x, y=None, **fit_params):
        return self.scaler.fit_transform(x, y, **fit_params)

    def transform(self, x):
        return self.scaler.transform(x)

    def save(self, sk_path):
        sk_save(self.scaler, sk_path)

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)

    @classmethod
    def load(cls, sk_path):
        return sk_load(sk_path)


class MinMaxScaler(BaseScaler):
    def __init__(self, **kwargs):
        super(MinMaxScaler, self).__init__(**kwargs)
        self.scaler = preprocessing.MinMaxScaler(**self.kwargs)


class StandardScaler(BaseScaler):
    def __init__(self, **kwargs):
        super(StandardScaler, self).__init__(**kwargs)
        self.scaler = preprocessing.StandardScaler(**self.kwargs)


class RobustScaler(BaseScaler):
    def __init__(self, **kwargs):
        super(RobustScaler, self).__init__(**kwargs)
        self.scaler = preprocessing.RobustScaler(**self.kwargs)


class MaxAbsScaler(BaseScaler):
    def __init__(self, **kwargs):
        super(MaxAbsScaler, self).__init__(**kwargs)
        self.scaler = preprocessing.MaxAbsScaler(**self.kwargs)


class Normalizer(BaseScaler):
    def __init__(self, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.scaler = preprocessing.Normalizer(**self.kwargs)


class Scaler(object):
    def __init__(self, scaler_type, **scaler_kwargs):
        self.scaler_type = scaler_type
        self.scaler_kwargs = scaler_kwargs
        self._process_fn()  # 归一化工具

    def _process_fn(self):
        if self.scaler_type in {"MinMaxScaler", "min-max"}:
            self.scaler = MinMaxScaler(**self.scaler_kwargs)
        elif self.scaler_type == {"StandardScaler", "standard"}:
            self.scaler = StandardScaler(**self.scaler_kwargs)
        elif self.scaler_type == {"MaxAbsScaler", "max-abs"}:
            self.scaler = MaxAbsScaler(**self.scaler_kwargs)
        elif self.scaler_type == {"RobustScaler", "robust"}:
            self.scaler = RobustScaler(**self.scaler_kwargs)
        elif self.scaler_type == {"Normalizer", "normalizer"}:
            self.scaler = Normalizer(**self.scaler_kwargs)
        else:
            raise ValueError("Scaler type not supported.")

    def fit(self, x, y=None, **fit_params):
        if self.scaler is not None:
            return self.scaler.fit(x, y, **fit_params)
            # return self.scaler
        else:
            return x

    def transform(self, x):
        if self.scaler is not None:
            return self.scaler.transform(x)
        else:
            return x

    def fit_transform(self, x, y=None, **fit_params):
        if self.scaler is not None:
            return self.scaler.fit_transform(x, y, **fit_params)
        else:
            return x

    def inverse_transform(self, x):
        if self.scaler is not None:
            return self.scaler.inverse_transform(x)
        else:
            return x

    def inverse_y_transform(self, y, feature_list, feature_idx):
        """
        :param y:
        :param feature_list: 特征list 或 特征数
        :param feature_idx:
        :return:
        """
        if isinstance(feature_list, list):
            dummy = np.zeros((len(y), len(feature_list)))
        else:
            dummy = np.zeros((len(y), feature_list))
        dummy[:, feature_idx] = y
        return self.scaler.inverse_transform(dummy)[:, feature_idx]

    def save(self, path):  # 保存模型
        if self.scaler is not None:
            self.scaler.save(path)

    @classmethod
    def load(cls, path):  # 加载模型
        return BaseScaler.load(path)


class FeatureSelector(object):  # 选择特征输出
    def __init__(self, feature_name, return_type='array'):
        self.feature_name = feature_name
        self.return_type = return_type

    def __call__(self, df):
        if self.return_type == 'array':
            return df[self.feature_name].values
        elif self.return_type == 'dataframe':
            return df[self.feature_name]
        elif self.return_type == 'list':
            return df[self.feature_name].tolist()
        elif self.return_type == 'dict':
            return df[self.feature_name].to_dict()
        elif self.return_type is None:
            return df[self.feature_name], df[self.feature_name].values
        else:
            raise ValueError("Return type not supported.")


class BuildSequenceFeature(object):  # 构建序列特征
    def __init__(self, feature_name, target_name=None, target_index=0, window_size=96*7, step_size=96):
        """
        :param feature_name: 特征名称
        :param target_name: 目标名称
        :param target_index: 目标索引
        :param window_size: 特征窗口大小，即序列长度
        :param step_size: 预测步长
        """
        self.feature_name = feature_name
        self.target_name = target_name
        self.target_index = target_index
        self.window_size = window_size
        self.step_size = step_size

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            x, y = [], []
            for i in range(len(data) - self.window_size - self.step_size):
                x.append(data[i:i+self.window_size])
                y.append(data[i+self.window_size:i+self.window_size+self.step_size, self.target_index])  # 只预测负荷值
            return np.array(x), np.array(y)

        elif isinstance(data, pd.DataFrame):
            x, y = [], []
            for i in range(len(data) - self.window_size - self.step_size):
                x.append(data.iloc[i:i+self.window_size, self.feature_name].values)
                y.append(data.iloc[i+self.window_size:i+self.window_size+self.step_size, self.target_name].values)  # 只预测负荷值
            return np.array(x), np.array(y)

        else:
            raise ValueError("Data type not supported.")


if __name__ == '__main__':
    # 测试
    # data = np.random.rand(10, 10)
    # bs = BuildSequenceFeature(feature_name=None, target_index=0, window_size=5, step_size=3)
    # x, y =bs(data)
    # print(x.shape, y.shape)
    data = np.random.rand(10, 10)
    print(data)
    # print("----------数据分开------------")
    # sc1 = Scaler("MinMaxScaler", feature_range=(0, 1))
    # sc2 = Scaler("MinMaxScaler", feature_range=(0, 1))
    # train_x, train_y = data[:5, 1:], data[:5, 0]
    # test_x, test_y = data[5:, 1:], data[5:, 0]
    # sc1 = sc1.fit(train_x)
    #
    # print("测试x")
    # print(train_x)
    # print("-----------------")
    # train_x_ = sc1.transform(train_x)
    # inverse_train_x = sc1.inverse_transform(train_x_)
    # print(inverse_train_x)
    # print("测试y")
    # sc2 = sc2.fit(np.reshape(train_y, (-1, 1)))
    # print(train_y)
    # print("-----------------")
    # train_y_ = sc2.transform(np.reshape(train_y, (-1, 1)))
    # inverse_train_y = sc2.inverse_transform(train_y_)
    # print(inverse_train_y)

    print("---------------数据不分开-----------------")
    sc3 = Scaler("MinMaxScaler", feature_range=(0, 1))
    train_data, test_data = data[:5, :], data[5:, :]
    print("---------------测试------------------")
    print("整体 训练集")
    print(train_data)
    sc3.fit(train_data)
    train_data_ = sc3.transform(train_data)
    print(train_data_)
    print(sc3.inverse_transform(train_data_))
    print("整体 训练集")
    print(test_data)
    test_data_ = sc3.transform(test_data)
    print(test_data_)
    print(sc3.inverse_transform(test_data_))
    print("---------------测试------------------")
    x = test_data[:, 1:]
    y = test_data[:, 0]
    print("单独 测试其中的y")
    print(sc3.inverse_y_transform(y, range(10), 0))
