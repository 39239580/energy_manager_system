from core.algo.utils import *


class CNNBiLSTM1(keras.Model):
    def __init__(self, layer, cnn_filters, cnn_kernel_size, cnn_strides, cnn_padding, cnn_rate, lstm_units,
                 lstm_dropout, lstm_recurrent_dropout, last_return_sequence,  pool_type, dnn_units, dnn_dropout,
                 num_heads, key_dim, final_units=1, name=None, **kwargs):
        super(CNNBiLSTM1, self).__init__(name=name, **kwargs)
        self.layer = layer
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_strides = cnn_strides
        self.cnn_padding = cnn_padding
        self.cnn_rate = cnn_rate
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.last_return_sequence = last_return_sequence
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.final_units = final_units
        self.mtd_layer = MultiTimeDistributedLayer(layers=self.layer)
        self.cnn_layer = CNNLayer2(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size, strides=self.cnn_strides,
                                   padding=self.cnn_padding, rate=self.cnn_rate)
        self.bi_lstm_layer = MultiBiLSTMLayer(units=self.lstm_units, dropout=self.lstm_dropout,
                                              recurrent_dropout=self.lstm_recurrent_dropout,
                                              last_return_sequence=self.last_return_sequence)
        if self.last_return_sequence:  # 返回为序列的时候，需要使用MultiHeadAttention
            self.attn_layer = MultiHeadAttentionLayer(head_num=self.num_heads, key_dim=self.key_dim, dropout=0.0)
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

    def call(self, inputs, training=None, **kwargs):
        mtd_output = self.mtd_layer(inputs)
        cnn_output = self.cnn_layer(mtd_output)
        lstm_output = self.bi_lstm_layer(cnn_output)
        if self.last_return_sequence:
            attention_output = self.attn_layer(lstm_output, lstm_output, training=training, **kwargs)
            if self.pool_type is not None:  # 池化层来降维度
                pool_output = []
                for pool in self.pool_layer:
                    pool_output.append(pool(attention_output))
                concat_output = self.concat_layer(pool_output)
            else:
                concat_output = self.flatten_layer(lstm_output)
            dnn_output = self.dnn_layer(concat_output)
        else:
            dnn_output = self.dnn_layer(lstm_output)
        output = self.output_layer(dnn_output)
        return output

    def get_config(self):
        config = super(CNNBiLSTM1, self).get_config()
        config.update(
            {
                'layer': self.layer,
                'cnn_filters': self.cnn_filters,
                'cnn_kernel_size': self.cnn_kernel_size,
                'cnn_strides': self.cnn_strides,
                'cnn_padding': self.cnn_padding,
                'cnn_rate': self.cnn_rate,
                'lstm_units': self.lstm_units,
                'lstm_dropout': self.lstm_dropout,
                'lstm_recurrent_dropout': self.lstm_recurrent_dropout,
                'last_return_sequence': self.last_return_sequence,
                'pool_type': self.pool_type,
                'dnn_units': self.dnn_units,
                'dnn_dropout': self.dnn_dropout,
                'num_heads': self.num_heads,
                'key_dim': self.key_dim,
                'final_units': self.final_units
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNNBiLSTM2(keras.Model):
    def __init__(self, layer, cnn_filters, cnn_kernel_size, cnn_strides, cnn_activation, cnn_padding, cnn_rate,
                 lstm_units, lstm_dropout, lstm_recurrent_dropout, last_return_sequence,  pool_type, dnn_units,
                 dnn_dropout, dnn_activation, num_heads, key_dim, final_units=1, name=None, **kwargs):
        super(CNNBiLSTM2, self).__init__(name=name, **kwargs)
        self.layer = layer
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_strides = cnn_strides
        self.cnn_activation = cnn_activation
        self.cnn_padding = cnn_padding
        self.cnn_rate = cnn_rate
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.last_return_sequence = last_return_sequence
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.final_units = final_units
        self.mtd_layer = MultiTimeDistributedLayer(layers=self.layer)
        self.cnn_layer = CNNLayer1(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size, strides=self.cnn_strides,
                                   activation=self.cnn_activation, padding=self.cnn_padding, rate=self.cnn_rate)
        self.bi_lstm_layer = MultiBiLSTMLayer(units=self.lstm_units, dropout=self.lstm_dropout,
                                              recurrent_dropout=self.lstm_recurrent_dropout)
        if self.last_return_sequence:  # 返回为序列的时候，需要使用MultiHeadAttention
            self.attn_layer = MultiHeadAttentionLayer(head_num=self.num_heads, key_dim=self.key_dim, dropout=0.0)
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
        self.dnn_layer = DNNLayer1(units=self.dnn_units, dropout=self.dnn_dropout, activation=self.dnn_activation)
        self.output_layer = OutputLayer(units=self.final_units)

    def call(self, inputs, training=None, **kwargs):
        mtd_output = self.mtd_layer(inputs)
        cnn_output = self.cnn_layer(mtd_output)
        lstm_output = self.bi_lstm_layer(cnn_output)
        if self.last_return_sequence:
            attention_output = self.attn_layer(lstm_output, lstm_output, training=training, **kwargs)
            if self.pool_type is not None:  # 池化层来降维度
                pool_output = []
                for pool in self.pool_layer:
                    pool_output.append(pool(attention_output))
                concat_output = self.concat_layer(pool_output)
            else:
                concat_output = self.flatten_layer(lstm_output)
            dnn_output = self.dnn_layer(concat_output)
        else:
            dnn_output = self.dnn_layer(lstm_output)
        output = self.output_layer(dnn_output)
        return output

    def get_config(self):
        config = super(CNNBiLSTM2, self).get_config()
        config.update(
            {
                'layer': self.layer,
                'cnn_filters': self.cnn_filters,
                'cnn_kernel_size': self.cnn_kernel_size,
                'cnn_activation': self.cnn_activation,
                'cnn_strides': self.cnn_strides,
                'cnn_padding': self.cnn_padding,
                'cnn_rate': self.cnn_rate,
                'lstm_units': self.lstm_units,
                'lstm_dropout': self.lstm_dropout,
                'lstm_recurrent_dropout': self.lstm_recurrent_dropout,
                'dnn_units': self.dnn_units,
                'dnn_activation': self.dnn_activation,
                'dnn_dropout': self.dnn_dropout,
                'num_heads': self.num_heads,
                'key_dim': self.key_dim,
                'final_units': self.final_units
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNNBiLSTM(keras.Model):
    def __init__(self, precess_units, cnn_filters, cnn_kernel_size, cnn_strides, cnn_activation, cnn_padding, cnn_rate,
                 lstm_units, lstm_dropout, lstm_recurrent_dropout, last_return_sequence,  pool_type,
                 dnn_units, dnn_dropout, dnn_activation, num_heads,
                 key_dim, final_units=1, model_no=0, name=None, **kwargs):
        super(CNNBiLSTM, self).__init__(name=name, **kwargs)
        self.precess_units = precess_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_strides = cnn_strides
        self.cnn_activation = cnn_activation
        self.cnn_padding = cnn_padding
        self.cnn_rate = cnn_rate
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.last_return_sequence = last_return_sequence
        self.pool_type = pool_type
        self.dnn_units = dnn_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.final_units = final_units
        self.model_no = model_no
        layer = []
        for i in range(len(precess_units)):
            layer.append(keras.layers.Dense(units=precess_units[i], activation='relu'))
        if self.model_no == 0:
            self.sub_model = CNNBiLSTM1(
                layer, self.cnn_filters, self.cnn_kernel_size, self.cnn_strides, self.cnn_padding, self.cnn_rate,
                self.lstm_units, self.lstm_dropout, self.lstm_recurrent_dropout, self.last_return_sequence,
                self.pool_type, self.dnn_units, self.dnn_dropout, self.num_heads, self.key_dim, self.final_units)
        else:
            self.sub_model = CNNBiLSTM2(
                layer, self.cnn_filters, self.cnn_kernel_size, self.cnn_strides, self.cnn_activation, self.cnn_padding,
                self.cnn_rate, self.lstm_units, self.lstm_dropout, self.lstm_recurrent_dropout,
                self.last_return_sequence,  self.pool_type, self.dnn_units, self.dnn_dropout, self.dnn_activation,
                self.num_heads, self.key_dim, self.final_units)

    def call(self, inputs, training=None, **kwargs):
        output = self.sub_model(inputs)
        return output

    def get_config(self):
        config = super(CNNBiLSTM, self).get_config()
        config.update(
            {
                'precess_units': self.precess_units,
                'cnn_filters': self.cnn_filters,
                'cnn_kernel_size': self.cnn_kernel_size,
                'cnn_activation': self.cnn_activation,
                'cnn_strides': self.cnn_strides,
                'cnn_padding': self.cnn_padding,
                'cnn_rate': self.cnn_rate,
                'lstm_units': self.lstm_units,
                'lstm_dropout': self.lstm_dropout,
                'lstm_recurrent_dropout': self.lstm_recurrent_dropout,
                'last_return_sequence': self.last_return_sequence,
                'pool_type': self.pool_type,
                'dnn_units': self.dnn_units,
                'dnn_activation': self.dnn_activation,
                'dnn_dropout': self.dnn_dropout,
                'num_heads': self.num_heads,
                'key_dim': self.key_dim,
                'final_units': self.final_units,
                'model_no': self.model_no
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



