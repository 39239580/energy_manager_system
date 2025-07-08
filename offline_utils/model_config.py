

class CNNBiLSTMConfig(object):
    process_units = [64, 32]
    cnn_filters = [64, 128, 256]
    cnn_kernel_size = [5, 3, 3]
    cnn_strides = [1, 1, 1]
    cnn_activation = ["relu", "relu", "relu"]
    cnn_padding = ["same", "same", "same"]
    cnn_rate = [0.2, 0.2, 0.2]
    lstm_units = [256, 256]
    lstm_dropout = [0.2, 0.2]
    lstm_recurrent_dropout = [0.1, 0.1]
    last_return_sequence = True,
    pool_type = ["global_max", "global_avg"]
    dnn_units = [512, 256]
    dnn_dropout = [0.3, 0.3]
    dnn_activation = ["relu", "relu"]
    num_heads = 4
    key_dim = 64
    final_units = 96


class DeepModelTrainerConfig(object):
    learning_rate = 0.001
    batch_size = 32
    epochs = 100
    optimizer = "adam"
    loss = "mse"
    metrics = ["accuracy"]
    verbose = 1
    callbacks = {
        "EarlyStopping": {"patience": 15},
        "ReduceLROnPlateau": {"patience": 5, "factor": 0.5},
        "TensorBoard": {"log_dir": "offline_logs/train.logs",
                        "histogram_freq": 1}}
