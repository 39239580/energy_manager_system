from offline_utils.data_preprocess_fn import DataProcessUtils
from core.algo.cnn_bi_lstm import CNNBiLSTM
from offline_utils.model_config import CNNBiLSTMConfig
from tensorflow import keras
from core.algo.utils import optimizer_config
from offline_utils.model_config import DeepModelTrainerConfig
from core.algo.aux_utils import get_quantile_metrics
from core.algo.callback_utils import DeepModelCallback


class DeepPipeline(object):
    def __init__(self, feature_name, test_ratio=0.2, validation_ratio=0.3, random_state=42, scaler_type="min-max",
                 target_name=None, target_index=0, window_size=96 * 7, step_size=96, model_name="cnn_bi_lstm"):
        self.feature_name = feature_name
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.target_name = target_name
        self.target_index = target_index
        self.window_size = window_size
        self.step_size = step_size
        self.model_name = model_name
        self.process_tools = self._process_fn()
        self.model = self._init_model()
        self.callbacks = DeepModelCallback()

    def _process_fn(self):
        return DataProcessUtils(
            feature_name=self.feature_name, test_ratio=self.test_ratio, random_state=self.random_state,
            scaler_type=self.scaler_type, target_name=self.target_name, target_index=self.target_index,
            window_size=self.window_size, step_size=self.step_size)

    def process_fit_transform(self, data):
        return self.process_tools(data)

    def process_validation(self, x, y):  # 划分验证集
        return self.process_tools.split_validation(x, y)

    def _init_model(self):
        if self.model_name == "cnn_bi_lstm":
            return CNNBiLSTM(precess_units=CNNBiLSTMConfig.process_units, cnn_filters=CNNBiLSTMConfig.cnn_filters,
                             cnn_kernel_size=CNNBiLSTMConfig.cnn_kernel_size, cnn_strides=CNNBiLSTMConfig.cnn_strides,
                             cnn_activation=CNNBiLSTMConfig.cnn_activation, cnn_padding=CNNBiLSTMConfig.cnn_padding,
                             cnn_rate=CNNBiLSTMConfig.cnn_rate, lstm_units=CNNBiLSTMConfig.lstm_units,
                             lstm_dropout=CNNBiLSTMConfig.lstm_dropout,
                             lstm_recurrent_dropout=CNNBiLSTMConfig.lstm_recurrent_dropout,
                             last_return_sequence=CNNBiLSTMConfig.last_return_sequence,
                             pool_type=CNNBiLSTMConfig.pool_type,
                             dnn_units=CNNBiLSTMConfig.dnn_units, dnn_dropout=CNNBiLSTMConfig.dnn_dropout,
                             dnn_activation=CNNBiLSTMConfig.dnn_activation, num_heads=CNNBiLSTMConfig.num_heads,
                             key_dim=CNNBiLSTMConfig.key_dim, final_units=CNNBiLSTMConfig.final_units)

    def _init_callbacks(self):
        callbacks = DeepModelTrainerConfig.callbacks
        if "EarlyStopping" in callbacks:
            self.callbacks.add_early_stopping_callback(**callbacks["EarlyStopping"])
        if "ReduceLROnPlateau" in callbacks:
            self.callbacks.add_reduce_lr_on_plateau_callback(**callbacks["ReduceLROnPlateau"])
        if "ModelCheckpoint" in callbacks:
            self.callbacks.add_model_checkpoint_callback(**callbacks["ModelCheckpoint"])
        if "TensorBoard" in callbacks:
            self.callbacks.add_tensorboard_callback(**callbacks["TensorBoard"])

    def compile_model(self, optimizer="adam", loss="mse", metrics=["mae"], use_quantile=True, quantile_q=0.1):
        if isinstance(optimizer, str):
            optimizer = optimizer_config(optimizer, learning_rate=DeepModelTrainerConfig.learning_rate)
        if use_quantile:
            metrics = get_quantile_metrics(quantile_q)
            metrics = list(metrics.values())
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary_model(self):
        return self.model.summary()

    def plot_model(self, to_file='model_architecture.png', show_shapes=True, show_layer_names=True):
        keras.utils.plot_model(self.model, to_file=to_file, show_shapes=show_shapes, show_layer_names=show_layer_names)

    def fit_model(self, x_train, y_train, x_val=None, y_val=None, batch_size=32, epochs=10, verbose=1, callbacks=None, **kwargs):
        if callbacks is None:
            callbacks = self.callbacks.callbacks
        return self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                              validation_data=(x_val, y_val), callbacks=callbacks, **kwargs)

    def scale_model_save(self, save_path="offline_models/scaler.pkl"):
        self.process_tools.save(save_path)

    def predict_model(self, x, batch_size=None, verbose="auto", **kwargs):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, **kwargs)

    def evaluate_model(self, x=None, y=None, batch_size=None, **kwargs):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, **kwargs)

    def save_model(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.model.save(filepath, overwrite=overwrite, save_format=save_format, **kwargs)




