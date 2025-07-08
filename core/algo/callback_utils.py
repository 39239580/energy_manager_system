from tensorflow import keras


def tensorboard_callback(log_dir="logs", histogram_freq=1, **kwargs):
    return keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, **kwargs)


def early_stopping_callback(monitor="val_loss", min_delta=0, patience=0, verbose=1, restore_best_weights=False,
                            start_from_epoch=0, **kwargs):
    return keras.callbacks.EarlyStopping(
        monitor=monitor, min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
        start_from_epoch=start_from_epoch, verbose=verbose, **kwargs)


def model_checkpoint_callback(filepath, monitor="val_loss", save_best_only=True, save_weights_only=True,
                              mode="auto", verbose=1, **kwargs):
    return keras.callbacks.ModelCheckpoint(
        filepath=filepath, monitor=monitor, save_best_only=save_best_only, save_weights_only=save_weights_only,
        mode=mode, verbose=verbose, **kwargs)


# keras.callbacks.ReduceLROnPlateau 是一种动态学习率调整策略，当监控指标不再改善时，学习率会减小。
def reduce_lr_on_plateau_callback(monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0,
                                  mode="auto", **kwargs):
    return keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=verbose,
                                             min_lr=min_lr, mode=mode, **kwargs)


class DeepModelCallback(object):  # 添加了一些回调函数的集合
    def __init__(self, callbacks=None):
        if callbacks is not None and not isinstance(callbacks, list):
            raise ValueError("callbacks must be a list")
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def add_tensorboard_callback(self, log_dir="logs", histogram_freq=1, **kwargs):
        self.add_callback(tensorboard_callback(log_dir=log_dir, histogram_freq=histogram_freq, **kwargs))

    def add_early_stopping_callback(self, monitor="val_loss", min_delta=0, patience=0, verbose=1, restore_best_weights=False,
                                    start_from_epoch=0, **kwargs):
        self.add_callback(early_stopping_callback(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose,
                                                  restore_best_weights=restore_best_weights, start_from_epoch=start_from_epoch,
                                                  **kwargs))

    def add_model_checkpoint_callback(self, filepath, monitor="val_loss", save_best_only=True, save_weights_only=True,
                                      mode="auto", verbose=1, **kwargs):
        self.add_callback(model_checkpoint_callback(filepath=filepath, monitor=monitor, save_best_only=save_best_only,
                                                    save_weights_only=save_weights_only, mode=mode, verbose=verbose,
                                                    **kwargs))

    def add_reduce_lr_on_plateau_callback(self, monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0,
                                          mode="auto", **kwargs):
        self.add_callback(reduce_lr_on_plateau_callback(monitor=monitor, factor=factor, patience=patience,
                                                        verbose=verbose,
                                                        min_lr=min_lr, mode=mode, **kwargs))

    def get_callbacks(self):
        return self.callbacks




