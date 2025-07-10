import tensorflow as tf
from tensorflow import keras
from core.algo.utils import DNNLayer1, DNNLayer2, OutputLayer


class DNN1(keras.Model):  # 非序列模型预测
    def __init__(self, units, activation, dropout, name=None, **kwargs):
        super(DNN1, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.layers = keras.Sequential()
        self.layers.add(DNNLayer1(units=self.units[:-1], activation=self.activation[:-1], dropout=self.dropout))
        self.layers.add(OutputLayer(units=self.units[-1], activation=self.activation[-1]))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(DNN1, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DNN2(keras.Model):  # 非序列模型预测
    def __init__(self, units, dropout, name=None, **kwargs):
        super(DNN2, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.layers = keras.Sequential(DNNLayer2(units=self.units[:-1], dropout=self.dropout))
        self.layers.add(OutputLayer(units=self.units[-1]))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(DNN2, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DNN3(keras.Model):  # 多步模型预测
    def __init__(self, units, activation, dropout, name=None, **kwargs):
        super(DNN3, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.layers = keras.Sequential()
        self.layers.add(keras.layers.Flatten())
        self.layers.add(DNNLayer1(units=self.units[:-1], activation=self.activation[:-1], dropout=self.dropout))
        self.layers.add(OutputLayer(units=self.units[-1], activation=self.activation[-1]))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(DNN3, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
