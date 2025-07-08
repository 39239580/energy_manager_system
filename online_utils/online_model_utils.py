from core.algo.model_tools import ModeInfer
from online_utils.model_config import DevelopMODELConfig


class OnlineModel(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = None
        self.model = None
        self.signature = None
        self.infer = None
        self.init_model()
        self.model = ModeInfer(model_path=self.model_path)

    def init_model(self):
        if self.model_name == "cnn_bi_lstm":
            self.model_path = DevelopMODELConfig.CNN_BI_LSTM_MODEL["MODEL_PATH"]
        elif self.model_name == "cnn_lstm":
            self.model_path = DevelopMODELConfig.CNN_LSTM_MODEL["MODEL_PATH"]
        elif self.model_name == "cnn_gru":
            self.model_path = DevelopMODELConfig.CNN_GRU_MODEL["MODEL_PATH"]
        elif self.model_name == "cnn_lstm_attention":
            self.model_path = DevelopMODELConfig.CNN_LSTM_ATTENTION_MODEL["MODEL_PATH"]
        elif self.model_name == "cnn_gru_attention":
            self.model_path = DevelopMODELConfig.CNN_GRU_ATTENTION_MODEL["MODEL_PATH"]
        else:
            raise ValueError("Invalid model name")

    def infer_data(self, input_data):
        return self.infer.infer(input_data)