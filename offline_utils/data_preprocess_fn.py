from core.data_process_utils.process_fn import FeatureSelector
from core.data_process_utils.dataset_split_utils import DatasetSplitUtils
from core.data_process_utils.process_fn import Scaler
from core.data_process_utils.process_fn import BuildSequenceFeature
from core.data_process_utils.process_fn import sk_load, sk_save
import pandas as pd
import os


class DataProcessUtils(object):  #  针对的序列数据处理工具类
    def __init__(self, feature_name, test_ratio=0.3, validation_ratio=0.2, random_state=42, scaler_type="min-max",
                 target_name=None, target_index=0, window_size=96 * 7, step_size=96):
        self.feature_name = feature_name
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.target_name = target_name
        self.target_index = target_index
        self.window_size = window_size
        self.step_size = step_size
        self._feature_selector()
        self._dataset_split()
        self._validation_split()
        self._scaler()
        self._build_sequence_feature()

    def _feature_selector(self):
        self.selector = FeatureSelector(feature_name=self.feature_name)

    @classmethod
    def feature_selection(cls, feature_name, data):  # 特征选择
        return FeatureSelector(feature_name=feature_name)(data)

    def _dataset_split(self):
        self.splitter = DatasetSplitUtils(test_ratio=self.test_ratio, random_state=self.random_state)

    @classmethod
    def dataset_split(cls, data, test_ratio=0.2, random_state=42):  # 划分数据集
        splitter = DatasetSplitUtils(test_ratio=test_ratio, random_state=random_state)
        return splitter.split_dataset(data)

    def _validation_split(self):  # 划分验证集
        self.validation_splitter = DatasetSplitUtils(test_ratio=self.validation_ratio, random_state=self.random_state)

    @classmethod
    def validation_split(cls, x, y, validation_ratio=0.2, random_state=42):  # 划分验证集
        splitter = DatasetSplitUtils(test_ratio=validation_ratio, random_state=random_state)
        return splitter.split_dataset(x, y)

    def split_dataset(self, data):  # 划分数据集
        if isinstance(data, pd.DataFrame):
            dataset = self.selector(data)
        else:
            dataset = data
        train_dataset, test_dataset = self.splitter.split_dataset(dataset)
        return train_dataset, test_dataset

    def split_validation(self, x, y, index=None):  # 划分验证集
        return self.splitter.split_dataset(x, y, index=index)

    def _scaler(self):
        self.scaler = Scaler(self.scaler_type)

    @classmethod
    def scaler_op(cls, data, scaler_type="min-max"):  # 标准化
        scaler = Scaler(scaler_type)
        scaler.fit(x=data)
        return scaler, scaler.transform(x=data)

    def _build_sequence_feature(self):
        self.sequence_feature = BuildSequenceFeature(feature_name=self.feature_name, target_name=self.target_name,
                                                     target_index=self.target_index, window_size=self.window_size,
                                                     step_size=self.step_size)

    @classmethod
    def build_sequence_feature(cls, train_scaler, test_scaler, feature_name=None, target_name=None, target_index=0,
                               window_size=96 * 7, step_size=96):  # 构建序列特征
        sequence_feature = BuildSequenceFeature(
            feature_name=feature_name, target_name=target_name, target_index=target_index, window_size=window_size,
            step_size=step_size)
        train_x, train_y = sequence_feature(train_scaler)
        if test_scaler is not None:
            test_x, test_y = sequence_feature(test_scaler)
            return train_x, train_y, test_x, test_y, sequence_feature
        else:
            return train_x, train_y, sequence_feature

    def build_sample(self, train_dataset, test_dataset=None):  # 构建样本
        train_scaler = self.scaler.fit_transform(train_dataset)
        train_x, train_y = self.sequence_feature(train_scaler)
        if test_dataset is not None:
            test_scaler = self.scaler.transform(test_dataset)
            test_x, test_y = self.sequence_feature(test_scaler)
            return train_x, train_y, test_x, test_y
        else:
            return train_x, train_y

    def save_scaler(self, model_path, model_name="/sequence_feature.pkl"):  # 保存模型
        self.scaler.save(os.path.join(model_path,  model_name))

    def save_sequence_feature(self, model_path, model_name="/scaler.pkl"):  # 保存序列特征处理器
        sk_save(self.sequence_feature, os.path.join(model_path, model_name))

    def save_data_preprocess(self, model_path, model_name="/sequence_feature.pkl"):  # 保存数据预处理器
        sk_save(self, os.path.join(model_path, model_name))

    def save_all(self, model_path, scaler_name="scaler.pkl", sequence_feature_name="sequence_feature.pkl",
                 data_preprocess_name="data_preprocess.pkl"):
        self.save_scaler(os.path.join(model_path, scaler_name))
        self.save_sequence_feature(os.path.join(model_path, sequence_feature_name))
        self.save_data_preprocess(os.path.join(model_path, data_preprocess_name))

    def __call__(self, data):
        dataset = self.selector(data)
        train_dataset, test_dataset = self.splitter.split_dataset(dataset)
        train_scaler = self.scaler.fit_transform(train_dataset)
        test_scaler = self.scaler.transform(test_dataset)
        train_x, train_y = self.sequence_feature(train_scaler)
        test_x, test_y = self.sequence_feature(test_scaler)
        return train_x, train_y, test_x, test_y, self.sequence_feature


class DataProcessUtilsLoder(object):   # 针对的序列数据处理工具类, 加载模型
    def __init__(self, model_path, scaler_name="scaler.pkl", sequence_feature_name="sequence_feature.pkl",
                 data_preprocess_name="data_preprocess.pkl", test_ratio=0.3, validation_ratio=0.2, random_state=42,
                 target_name=None, target_index=0, window_size=96 * 7, step_size=96):
        self.model_path = model_path
        self.scaler_name = scaler_name
        self.sequence_feature_name = sequence_feature_name
        self.data_preprocess_name = data_preprocess_name
        if self.data_preprocess_name is not None:
            self.data_preprocess = self._load_data_preprocess()
            self.scaler = self.data_preprocess.scaler
            self.sequence_feature = self.data_preprocess.sequence_feature
            self.selector = self.data_preprocess.selector
            self.splitter = self.data_preprocess.splitter
            self.validation_splitter = self.data_preprocess.validation_splitter
            if test_ratio is not None:
                self.data_preprocess.test_ratio = test_ratio
            if validation_ratio is not None:
                self.data_preprocess.validation_ratio = validation_ratio
            if random_state is not None:
                self.data_preprocess.random_state = random_state
            if target_name is not None:
                self.data_preprocess.target_name = target_name
            if target_index is not None:
                self.data_preprocess.target_index = target_index
            if window_size is not None:
                self.data_preprocess.window_size = window_size
            if step_size is not None:
                self.data_preprocess.step_size = step_size
        else:
            self.data_preprocess = None
            self.selector = None
            if scaler_name is not None:
                self.scaler = self._load_scaler()
            else:
                self.scaler = None
            if sequence_feature_name is not None:
                self.sequence_feature = self._load_sequence_feature()
            else:
                self.sequence_feature = None

    def load_scaler(self, model_path=None, scaler_name="/scaler.pkl"):  # 加载模型
        if (model_path is not None) and (scaler_name is not None):
            self.model_path = model_path
            self.scaler_name = scaler_name
        self.scaler = self._load_scaler()
        return self.scaler

    def _load_scaler(self):  # 加载模型
        return sk_load(os.path.join(self.model_path + self.scaler_name))

    def load_sequence_feature(self, model_path=None, sequence_feature_name="/sequence_feature.pkl"):  # 加载序列特征处理器
        if (model_path is not None) and (sequence_feature_name is not None):
            self.model_path = model_path
            self.sequence_feature_name = sequence_feature_name
        self.sequence_feature = self._load_sequence_feature()
        return self.sequence_feature

    def _load_sequence_feature(self):  # 加载序列特征处理器
        return sk_load(os.path.join(self.model_path , self.sequence_feature_name))

    def load_data_preprocess(self, model_path=None, data_preprocess_name="/data_preprocess.pkl"):  # 加载数据预处理器
        if (model_path is not None) and (data_preprocess_name is not None):
            self.model_path = model_path
            self.data_preprocess_name = data_preprocess_name
        self.data_preprocess = self._load_data_preprocess()
        return self.data_preprocess

    def _load_data_preprocess(self):  # 加载数据预处理器
        return sk_load(os.path.join(self.model_path, self.data_preprocess_name))

    def selector(self, data):  # 特征选择
        return self.selector(data)

    def scaler_transform(self, data):  # 标准化
        return self.scaler.transform(data)

    def scaler_fit_transform(self, data):  # 标准化
        return self.scaler.fit_transform(data)

    def scaler_inverse_transform(self, data):  # 反标准化
        return self.scaler.inverse_transform(data)

    def scaler_inverse_transform_y(self, data, feature_list, feature_idx):  # 反标准化
        return self.scaler.inverse_y_transform(data, feature_list, feature_idx)

    def split_dataset(self, data, test_ratio=None, random_state=None):  # 划分数据集
        if test_ratio is not None:
            self.data_preprocess.test_ratio = test_ratio
        if random_state is not None:
            self.data_preprocess.random_state = random_state
        return self.data_preprocess.split_dataset(data)

    def split_validation(self, x, y, index=None, validation_ratio=None, random_state=None):  # 划分验证集
        if validation_ratio is not None:
            self.data_preprocess.validation_ratio = validation_ratio
        if random_state is not None:
            self.data_preprocess.random_state = random_state

        return self.data_preprocess.split_validation(x, y, index=index)

    def build_sequence_feature(self, train_dataset, test_dataset, feature_name=None, target_name=None, target_index=None,    # 构建序列特征
                               window_size=None, step_size=None):
        if feature_name is not None:
            self.data_preprocess.feature_name = feature_name
        if target_name is not None:
            self.data_preprocess.target_name = target_name
        if target_index is not None:
            self.data_preprocess.target_index = target_index
        if window_size is not None:
            self.data_preprocess.window_size = window_size
        if step_size is not None:
            self.data_preprocess.step_size = step_size

        return self.data_preprocess.build_sequence_feature(train_dataset, test_dataset)

    def build_sample(self, train_dataset, test_dataset):  # 构建样本
        return self.data_preprocess.build_sample(train_dataset, test_dataset)






