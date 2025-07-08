import tensorflow as tf
from tensorflow import keras


class CNNPartLayer1(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, rate=0.2, name=None, **kwargs):
        super(CNNPartLayer1, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.rate = rate
        self.layer = keras.Sequential()
        self.layer.add(keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                                           strides=self.strides, padding=self.padding,
                                           activation=self.activation))
        self.layer.add(keras.layers.BatchNormalization())
        self.layer.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
        self.layer.add(keras.layers.Dropout(rate=self.rate))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(CNNPartLayer1, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides,
                       "padding": self.padding, "activation": self.activation, "rate": self.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNNPartLayer2(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, rate=0.2, name=None, **kwargs):
        super(CNNPartLayer2, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rate = rate
        self.layer = keras.Sequential()
        self.layer.add(keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                                           strides=self.strides, padding=self.padding))
        self.layer.add(keras.layers.BatchNormalization())
        self.layer.add(keras.layers.ReLU())
        self.layer.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
        self.layer.add(keras.layers.Dropout(rate=self.rate))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(CNNPartLayer2, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides,
                       "padding": self.padding, "rate": self.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNNLayer1(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, rate, name=None, **kwargs):
        super(CNNLayer1, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.rate = rate
        self.layer = keras.Sequential()
        for i in range(len(self.filters)):
            self.layer.add(CNNPartLayer1(filters=self.filters[i], kernel_size=self.kernel_size[i],
                                         strides=self.strides[i], padding=self.padding[i],
                                         activation=self.activation[i], rate=self.rate[i]))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(CNNLayer1, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides,
                       "padding": self.padding, "activation": self.activation, "rate": self.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNNLayer2(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, rate, name=None, **kwargs):
        super(CNNLayer2, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rate = rate
        self.layer = keras.Sequential()
        for i in range(len(self.filters)):
            self.layer.add(CNNPartLayer2(filters=self.filters[i], kernel_size=self.kernel_size[i],
                                         strides=self.strides[i], padding=self.padding[i],
                                         rate=self.rate[i]))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(CNNLayer2, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides,
                       "padding": self.padding, "rate": self.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DNNLayer1(keras.layers.Layer):
    def __init__(self, units, activation, dropout, name=None, **kwargs):
        super(DNNLayer1, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.layer = keras.Sequential()
        for i in range(len(self.units)):
            self.layer.add(keras.layers.Dense(units=self.units[i], activation=self.activation[i]))
            self.layer.add(keras.layers.BatchNormalization())
            self.layer.add(keras.layers.Dropout(rate=self.dropout[i]))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(DNNLayer1, self).get_config()
        config.update({"units": self.units, "activation": self.activation, "dropout": self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DNNLayer2(keras.layers.Layer):
    def __init__(self, units, dropout, name=None, **kwargs):
        super(DNNLayer2, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.layer = keras.Sequential()
        for i in range(len(self.units)):
            self.layer.add(keras.layers.Dense(units=self.units[i]))
            self.layer.add(keras.layers.BatchNormalization())
            self.layer.add(keras.layers.ReLU())
            self.layer.add(keras.layers.Dropout(rate=self.dropout[i]))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(DNNLayer2, self).get_config()
        config.update({"units": self.units, "dropout": self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class OutputLayer(keras.layers.Layer):
    def __init__(self, units, activation="linear", name=None, **kwargs):
        super(OutputLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.layer = keras.Sequential()
        self.layer.add(keras.layers.Dense(units=self.units, activation=self.activation))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(OutputLayer, self).get_config()
        config.update({"units": self.units, "activation": self.activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BiLSTMLayer(keras.layers.Layer):
    def __init__(self, units, dropout, recurrent_dropout, name=None, **kwargs):
        super(BiLSTMLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.layer = keras.Sequential()
        self.layer.add(keras.layers.Bidirectional(keras.layers.LSTM(units=self.units, return_sequences=True,
                                                                    dropout=self.dropout,
                                                                    recurrent_dropout=self.recurrent_dropout)))
        self.layer.add(keras.layers.LayerNormalization())
        self.layer.add(keras.layers.Bidirectional(keras.layers.LSTM(units=self.units, dropout=self.dropout,
                                                                    return_sequences=self.recurrent_dropout)))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(BiLSTMLayer, self).get_config()
        config.update({"units": self.units, "dropout": self.dropout, "recurrent_dropout": self.recurrent_dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiBiLSTMLayer(keras.layers.Layer):
    def __init__(self, units, dropout, recurrent_dropout, last_return_sequence=True, name=None, **kwargs):
        super(MultiBiLSTMLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.last_return_sequence = last_return_sequence
        self.layer = keras.Sequential()
        for i in range(len(self.units)):
            if i != len(self.units)-1:
                self.layer.add(keras.layers.Bidirectional(keras.layers.LSTM(
                    units=self.units[i], dropout=self.dropout[i],
                    recurrent_dropout=self.recurrent_dropout[i],
                    return_sequences=True)))
                self.layer.add(keras.layers.LayerNormalization())
            else:
                self.layer.add(keras.layers.Bidirectional(keras.layers.LSTM(
                    units=self.units[i], dropout=self.dropout[i],
                    recurrent_dropout=self.recurrent_dropout[i],
                    return_sequences=self.last_return_sequence)))
                self.layer.add(keras.layers.LayerNormalization())

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(MultiBiLSTMLayer, self).get_config()
        config.update({"units": self.units, "dropout": self.dropout,
                       "recurrent_dropout": self.recurrent_dropout,
                       "last_return_sequence": self.last_return_sequence})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiTimeDistributedLayer(keras.layers.Layer):
    def __init__(self, layers, name=None, **kwargs):
        # 核心作用
        # TimeDistributed 允许你在保持时间步结构的同时，对每个时间步单独应用相同的层操作。它解决了以下关键问题：
        # 维度兼容性：将设计用于静态数据的层（如 Dense）应用于序列数据
        # 独立处理：确保每个时间步被独立处理，而不是跨时间步混，会导致维度不匹配错误。
        # 维度保留：保持输入张量的时间维度结构
        # 输入输出结构
        # 类型	输入形状	输出形状
        # 非时序层	(batch_size, features)	(batch_size, units)
        # TimeDistributed包装后	(batch_size, timesteps, features)	(batch_size, timesteps, units)
        super(MultiTimeDistributedLayer, self).__init__(name=name, **kwargs)
        self.layers = layers
        self.layer = keras.Sequential()
        for layer in self.layers:
            self.layer.add(keras.layers.TimeDistributed(layer))

    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(MultiTimeDistributedLayer, self).get_config()
        config.update({"layer": self.layer})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResidualLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(ResidualLayer, self).__init__(name=name, **kwargs)
        self.add_layer = keras.layers.Add()
        self.layer = keras.layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        add_output = self.add_layer(inputs, *args, **kwargs)
        return self.layer(add_output, *args, **kwargs)


class MultiHeadAttentionLayer(keras.layers.Layer):
    def __init__(self, head_num, key_dim, dropout=0.0, name=None, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(name=name, **kwargs)
        self.head_num = head_num
        self.key_dim = key_dim
        self.dropout = dropout
        self.mha = keras.layers.MultiHeadAttention(num_heads=self.head_num, key_dim=self.key_dim, dropout=self.dropout)
        self.res_layer = ResidualLayer()

    def call(self, inputs, *args, **kwargs):
        attention_output = self.mha(inputs, inputs, *args, **kwargs)
        return self.res_layer([inputs, attention_output])

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"head_num": self.head_num, "key_dim": self.key_dim, "dropout": self.dropout})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PoolingLayer(keras.layers.Layer):
    def __init__(self, pooling_type, name=None, **kwargs):
        super(PoolingLayer, self).__init__(name=name, **kwargs)
        self.pooling_type = pooling_type
        if self.pooling_type in ['global_avg', 'Global_Avg', 'GlobalAvg']:
            self.pooling_layer = keras.layers.GlobalAveragePooling1D()
        elif self.pooling_type in ['global_max', 'Global_Max', 'GlobalMax']:
            self.pooling_layer = keras.layers.GlobalMaxPooling1D()
        elif self.pooling_type in ['avg', 'AVG', 'Average', 'average']:
            self.pooling_layer = keras.layers.AveragePooling1D()
        elif self.pooling_type in ['max', 'MAX', 'Maximum', 'maximum']:
            self.pooling_layer = keras.layers.MaxPooling1D()
        else:
            raise ValueError('Unsupported pooling type: %s' % self.pooling_type)

    def call(self, inputs, *args, **kwargs):
        return self.pooling_layer(inputs, *args, **kwargs)

    def get_config(self):
        config = super(PoolingLayer, self).get_config()
        config.update({"pooling_type": self.pooling_type})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def optimizer_config(optimizer="adam", learning_rate=0.001):
    if optimizer == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        return keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == 'adamax':
        return keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == 'nadam':
        return keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == 'ftrl':
        return keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adamw':
        return keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        raise ValueError('Unsupported optimizer: %s' % optimizer)
