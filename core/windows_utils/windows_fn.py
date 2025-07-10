import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class WindowsGenerator(object):
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df,
                 label_columns=None):
        """
        :param input_width:  序列长度
        :param label_width: 标签的长度，即预测的步长
        :param shift: 偏移量
        :param train_df: 训练集dataframe
        :param val_df: 验证集dataframe
        :param test_df: 测试集dataframe
        :param label_columns: 标签列名列表
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns

        # 标签列的索引
        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.total_window_size = self.input_width + self.shift

        self.input_slice = slice(0, input_width)  # 输入序列的切片， 即从0 到 input_width 的切片
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]   # 生成特征的索引

        self.label_start = self.total_window_size - self.label_width  # 标签开始的索引
        self.labels_slice = slice(self.label_start, None)  # 标签的索引
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]  # 生成标签的索引

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """
        将一个序列切分成输入序列和标签序列
        :param features: 一个序列  feature 是 一个三维数组， 第一维是样本数，第二维是序列长度，第三维是特征数
        :return: 输入序列和标签序列
        """
        inputs = features[:, self.input_slice, :]   #
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)  # 标签的列名索引
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        """
        生成数据集
        :param data: [time_step, feature_num], 即数据长度为 time_step， 特征数为 feature_num， 现在需要构造序列样本
        其中None, sequence_length, feature_num
        :return: 数据集
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(  # 将数据集转成序列数据
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()







