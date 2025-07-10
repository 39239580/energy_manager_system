from core.algo.utils import MultiBiGRULayer, MultiBiLSTMLayer, PoolingLayer
from core.algo.utils import DNNLayer1, DNNLayer2, OutputLayer
from tensorflow import keras
import tensorflow as tf


class MultiBiLSTM1(keras.Model):
    def __init__(self, units, dropout, recurrent_dropout, pool_type, dnn_units, dnn_dropout, final_units,
                 last_return_sequence=True, name=None, **kwargs):
        super(MultiBiLSTM1, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.final_units = final_units
        self.last_return_sequence = last_return_sequence
        self.layer = keras.Sequential()
        self.bi_lstm_layer = MultiBiLSTMLayer(units=self.units, dropout=self.dropout, recurrent_dropout=self.recurrent,
                                              last_return_sequence=self.last_return_sequence)
        if self.last_return_sequence:  # 返回为序列的时候，需要使用MultiHeadAttention
            self.pool_layer = []
            if self.pool_type is not None:  # 池化层来降维度
                if isinstance(self.pool_type, str):
                    self.pool_layer.append(PoolingLayer(pooling_type=self.pool_type))
                else:
                    for pool in self.pool_type:
                        self.pool_layer.append(PoolingLayer(pooling_type=pool))
                self.concat_layer = keras.layers.Concatenate(axis=-1)
            else:  # 池化层为空，使用展平层
                self.flatten_layer = keras.layers.Flatten()
        self.dnn_layer = DNNLayer2(units=self.dnn_units, dropout=self.dnn_dropout)
        self.output_layer = OutputLayer(units=self.final_units)

    def call(self, inputs, training=None, mask=None):
        lstm_output = self.bi_lstm_layer(inputs, training=training, mask=mask)
        if self.last_return_sequence:
            if self.pool_type is not None:  # 池化层来降维度
                pool_output = []
                for pool in self.pool_layer:
                    pool_output.append(pool(lstm_output))
                concat_output = self.concat_layer(pool_output)
            else:
                concat_output = self.flatten_layer(lstm_output)
            dnn_output = self.dnn_layer(concat_output)
        else:
            dnn_output = self.dnn_layer(lstm_output)
        output = self.output_layer(dnn_output)
        return output

    def get_config(self):
        config = super(MultiBiLSTM1, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'pool_type': self.pool_type,
            'dnn_units': self.dnn_units,
            'dnn_dropout': self.dnn_dropout,
            'final_units': self.final_units,
            'last_return_sequence': self.last_return_sequence,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiBiLSTM2(keras.Model):
    def __init__(self, units, dropout, recurrent_dropout, pool_type, dnn_units, dnn_dropout, dnn_activation,
                 final_units, last_return_sequence=True, name=None, **kwargs):
        super(MultiBiLSTM2, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.final_units = final_units
        self.last_return_sequence = last_return_sequence
        self.layer = keras.Sequential()
        self.bi_lstm_layer = MultiBiLSTMLayer(units=self.units, dropout=self.dropout, recurrent_dropout=self.recurrent,
                                              last_return_sequence=self.last_return_sequence)
        if self.last_return_sequence:  # 返回为序列的时候，需要使用MultiHeadAttention
            self.pool_layer = []
            if self.pool_type is not None:  # 池化层来降维度
                if isinstance(self.pool_type, str):
                    self.pool_layer.append(PoolingLayer(pooling_type=self.pool_type))
                else:
                    for pool in self.pool_type:
                        self.pool_layer.append(PoolingLayer(pooling_type=pool))
                self.concat_layer = keras.layers.Concatenate(axis=-1)
            else:  # 池化层为空，使用展平层
                self.flatten_layer = keras.layers.Flatten()
        self.dnn_layer = DNNLayer1(units=self.dnn_units, dropout=self.dnn_dropout, activation=self.activation)
        self.output_layer = OutputLayer(units=self.final_units)

    def call(self, inputs, training=None, mask=None):
        lstm_output = self.bi_lstm_layer(inputs, training=training, mask=mask)
        if self.last_return_sequence:
            if self.pool_type is not None:  # 池化层来降维度
                pool_output = []
                for pool in self.pool_layer:
                    pool_output.append(pool(lstm_output))
                concat_output = self.concat_layer(pool_output)
            else:
                concat_output = self.flatten_layer(lstm_output)
            dnn_output = self.dnn_layer(concat_output)
        else:
            dnn_output = self.dnn_layer(lstm_output)
        output = self.output_layer(dnn_output)
        return output

    def get_config(self):
        config = super(MultiBiLSTM2, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'pool_type': self.pool_type,
            'dnn_units': self.dnn_units,
            'dnn_dropout': self.dnn_dropout,
            'dnn_activation': self.dnn_activation,
            'final_units': self.final_units,
            'last_return_sequence': self.last_return_sequence,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiBiGRU1(keras.Model):
    def __init__(self, units, dropout, recurrent_dropout, pool_type, dnn_units, dnn_dropout, final_units,
                 last_return_sequence=True, name=None, **kwargs):
        super(MultiBiGRU1, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.final_units = final_units
        self.last_return_sequence = last_return_sequence
        self.layer = keras.Sequential()
        self.bi_gru_layer = MultiBiGRULayer(units=self.units, dropout=self.dropout, recurrent_dropout=self.recurrent,
                                            last_return_sequence=self.last_return_sequence)
        if self.last_return_sequence:  # 返回为序列的时候，需要使用MultiHeadAttention
            self.pool_layer = []
            if self.pool_type is not None:  # 池化层来降维度
                if isinstance(self.pool_type, str):
                    self.pool_layer.append(PoolingLayer(pooling_type=self.pool_type))
                else:
                    for pool in self.pool_type:
                        self.pool_layer.append(PoolingLayer(pooling_type=pool))
                self.concat_layer = keras.layers.Concatenate(axis=-1)
            else:  # 池化层为空，使用展平层
                self.flatten_layer = keras.layers.Flatten()
        self.dnn_layer = DNNLayer2(units=self.dnn_units, dropout=self.dnn_dropout)
        self.output_layer = OutputLayer(units=self.final_units)

    def call(self, inputs, training=None, mask=None):
        gru_output = self.bi_gru_layer(inputs, training=training, mask=mask)
        if self.last_return_sequence:
            if self.pool_type is not None:  # 池化层来降维度
                pool_output = []
                for pool in self.pool_layer:
                    pool_output.append(pool(gru_output))
                concat_output = self.concat_layer(pool_output)
            else:
                concat_output = self.flatten_layer(gru_output)
            dnn_output = self.dnn_layer(concat_output)
        else:
            dnn_output = self.dnn_layer(gru_output)
        output = self.output_layer(dnn_output)
        return output

    def get_config(self):
        config = super(MultiBiGRU1, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'pool_type': self.pool_type,
            'dnn_units': self.dnn_units,
            'dnn_dropout': self.dnn_dropout,
            'final_units': self.final_units,
            'last_return_sequence': self.last_return_sequence,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiBiGRU2(keras.Model):
    def __init__(self, units, dropout, recurrent_dropout, pool_type, dnn_units, dnn_dropout, dnn_activation,
                 final_units, last_return_sequence=True, name=None, **kwargs):
        super(MultiBiGRU2, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.final_units = final_units
        self.last_return_sequence = last_return_sequence
        self.layer = keras.Sequential()
        self.bi_gru_layer = MultiBiGRULayer(units=self.units, dropout=self.dropout, recurrent_dropout=self.recurrent,
                                             last_return_sequence=self.last_return_sequence)
        if self.last_return_sequence:  # 返回为序列的时候，需要使用MultiHeadAttention
            self.pool_layer = []
            if self.pool_type is not None:  # 池化层来降维度
                if isinstance(self.pool_type, str):
                    self.pool_layer.append(PoolingLayer(pooling_type=self.pool_type))
                else:
                    for pool in self.pool_type:
                        self.pool_layer.append(PoolingLayer(pooling_type=pool))
                self.concat_layer = keras.layers.Concatenate(axis=-1)
            else:  # 池化层为空，使用展平层
                self.flatten_layer = keras.layers.Flatten()
        self.dnn_layer = DNNLayer1(units=self.dnn_units, dropout=self.dnn_dropout, activation=self.activation)
        self.output_layer = OutputLayer(units=self.final_units)

    def call(self, inputs, training=None, mask=None):
        gru_output = self.bi_gru_layer(inputs, training=training, mask=mask)
        if self.last_return_sequence:
            if self.pool_type is not None:  # 池化层来降维度
                pool_output = []
                for pool in self.pool_layer:
                    pool_output.append(pool(gru_output))
                concat_output = self.concat_layer(pool_output)
            else:
                concat_output = self.flatten_layer(gru_output)
            dnn_output = self.dnn_layer(concat_output)
        else:
            dnn_output = self.dnn_layer(gru_output)
        output = self.output_layer(dnn_output)
        return output

    def get_config(self):
        config = super(MultiBiGRU2, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'pool_type': self.pool_type,
            'dnn_units': self.dnn_units,
            'dnn_dropout': self.dnn_dropout,
            'dnn_activation': self.dnn_activation,
            'final_units': self.final_units,
            'last_return_sequence': self.last_return_sequence,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)