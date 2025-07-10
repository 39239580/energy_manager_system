from core.algo.utils import CNNLayer1, OutputLayer, CNNLayer2, DNNLayer1, DNNLayer2
from tensorflow import keras


class CNN1(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, activation, rate, dnn_units, dnn_activation, dnn_dropout,
                 final_units, final_activation, name=None, **kwargs):
        super(CNN1, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.rate = rate
        self.dnn_units = dnn_units
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.final_units = final_units
        self.final_activation = final_activation
        self.layers = keras.Sequential()
        self.layers.add(CNNLayer1(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                  padding=self.padding, activation=self.activation, rate=self.rate))

        self.layers.add(DNNLayer1(units=self.dnn_units, activation=self.dnn_activation, dropout=self.dnn_dropout))

        self.layers.add(OutputLayer(units=self.final_units, activation=self.final_activation))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(CNN1, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'rate': self.rate,
            'dnn_units': self.dnn_units,
            'dnn_activation': self.dnn_activation,
            'dnn_dropout': self.dnn_dropout,
            'final_units': self.final_units,
            'final_activation': self.final_activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNN2(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, rate, dnn_units, dnn_activation, dnn_dropout,
                 final_units, final_activation, name=None, **kwargs):
        super(CNN2, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rate = rate
        self.dnn_units = dnn_units
        self.dnn_activation = dnn_activation
        self.dnn_dropout = dnn_dropout
        self.final_units = final_units
        self.final_activation = final_activation
        self.layers = keras.Sequential()
        self.layers.add(CNNLayer2(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                  padding=self.padding, rate=self.rate))
        self.layers.add(DNNLayer1(units=self.dnn_units, activation=self.final_activation, dropout=self.dnn_dropout))
        self.layers.add(OutputLayer(units=self.final_units, activation=self.final_activation))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(CNN2, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'rate': self.rate,
            'dnn_units': self.dnn_units,
            'dnn_activation': self.dnn_activation,
            'dnn_dropout': self.dnn_dropout,
            'final_units': self.final_units,
            'final_activation': self.final_activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNN3(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, activation, rate, dnn_units, dnn_dropout,
                 final_units, final_activation, name=None, **kwargs):
        super(CNN3, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.rate = rate
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.final_units = final_units
        self.final_activation = final_activation
        self.layers = keras.Sequential()
        self.layers.add(CNNLayer1(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                  padding=self.padding, activation=self.activation, rate=self.rate))

        self.layers.add(DNNLayer2(units=self.dnn_units, dropout=self.dnn_dropout))

        self.layers.add(OutputLayer(units=self.final_units, activation=self.final_activation))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(CNN3, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'rate': self.rate,
            'dnn_units': self.dnn_units,
            'dnn_dropout': self.dnn_dropout,
            'final_units': self.final_units,
            'final_activation': self.final_activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNN4(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, rate, dnn_units, dnn_dropout,
                 final_units, final_activation, name=None, **kwargs):
        super(CNN4, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rate = rate
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.final_units = final_units
        self.final_activation = final_activation
        self.layers = keras.Sequential()
        self.layers.add(CNNLayer2(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                                  padding=self.padding, rate=self.rate))
        self.layers.add(DNNLayer2(units=self.dnn_units, dropout=self.dnn_dropout))
        self.layers.add(OutputLayer(units=self.final_units, activation=self.final_activation))

    def call(self, inputs, training=None, mask=None):
        return self.layers(inputs, training=training, mask=mask)

    def get_config(self):
        config = super(CNN2, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'rate': self.rate,
            'dnn_units': self.dnn_units,
            'dnn_dropout': self.dnn_dropout,
            'final_units': self.final_units,
            'final_activation': self.final_activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
