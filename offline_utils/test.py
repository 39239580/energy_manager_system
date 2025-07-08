from core.data_process_utils.process_fn import FeatureSelector
from core.data_process_utils.dataset_split_utils import DatasetSplitUtils
from core.data_process_utils.process_fn import Scaler
from core.data_process_utils.process_fn import BuildSequenceFeature
from core.algo.cnn_bi_lstm import CNNBiLSTM1


class DataProcessUtils(object):
    def __init__(self, feature_name, test_ratio=0.2, random_state=42, scaler_type="min-max", target_name=None,
                 target_index=0, window_size=96*7, step_size=96):
        self.feature_name = feature_name
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.target_name = target_name
        self.target_index = target_index
        self.window_size = window_size
        self.step_size = step_size
        self._feature_selector()
        self._dataset_split()
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

    def split_dataset(self, data):  # 划分数据集
        dataset = self.selector(data)
        train_dataset, test_dataset = self.splitter.split_dataset(dataset)
        return train_dataset, test_dataset

    def _scaler(self):
        self.scaler = Scaler(self.scaler_type)

    @classmethod
    def scaler_op(cls, data, scaler_type="min-max"):  # 标准化
        scaler = Scaler(scaler_type)
        scaler.fit(x=data)
        return scaler, scaler.transform(x=data)

    def _build_sequence_feature(self):
        self.sequence_feature = BuildSequenceFeature(feature_name=self.feature_name,target_name=self.target_name,
                                                     target_index=self.target_index, window_size=self.window_size,
                                                     step_size=self.step_size)

    @classmethod
    def build_sequence_feature(cls, train_scaler, test_scaler, feature_name, target_name=None, target_index=0,
                               window_size=96*7, step_size=96):  # 构建序列特征
        sequence_feature = BuildSequenceFeature(
            feature_name=feature_name, target_name=target_name, target_index=target_index, window_size=window_size,
            step_size=step_size)
        train_x, train_y = sequence_feature(train_scaler)
        test_x, test_y = sequence_feature(test_scaler)
        return train_x, train_y, test_x, test_y, sequence_feature

    def build_sample(self, train_dataset, test_dataset):  # 构建样本
        train_scaler = self.scaler.fit_transform(train_dataset)
        test_scaler = self.scaler.transform(test_dataset)
        train_x, train_y = self.sequence_feature(train_scaler)
        test_x, test_y = self.sequence_feature(test_scaler)
        return train_x, train_y, test_x, test_y

    def save(self, model_path):  # 保存模型
        self.scaler.save(model_path)

    def __call__(self, data):
        dataset = self.selector(data)
        train_dataset, test_dataset = self.splitter.split_dataset(dataset)
        train_scaler = self.scaler.fit_transform(train_dataset)
        test_scaler = self.scaler.transform(test_dataset)
        train_x, train_y = self.sequence_feature(train_scaler)
        test_x, test_y = self.sequence_feature(test_scaler)
        return train_x, train_y, test_x, test_y, self.sequence_feature



