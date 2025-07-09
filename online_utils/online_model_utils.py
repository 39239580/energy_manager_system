from core.algo.model_tools import ModeInfer
from online_utils.model_config import DevelopMODELConfig
import tensorflow as tf


class OnlineModel(object):
    def __init__(self, model_name, signature_flag="serving_default"):
        self.model_name = model_name
        self.signature_flag = signature_flag
        self.model_path = None
        self.model = None
        self.signature = None
        self.init_model()
        self.model = ModeInfer(model_path=self.model_path, signature_flag=self.signature_flag)
        self.infer = self.model.infer

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
        return self.model.infer_data(input_data)

    def get_input_tensor_name(self):
        return list(self.infer.structured_input_signature[1].keys())[0]

    # def get_output_tensor_name(self):
    #     return list(self.infer.structured_outputs.keys())[0]

    def get_input_shape(self):
        input_tensor = next(iter(self.infer.structured_input_signature[1].values()))
        # 排除批次维度
        return tuple(dim for dim in input_tensor.shape.as_list()[1:] if dim is not None)
        # return self.infer.structured_input_signature[1]

    def get_output_tensor_name(self):
        """获取模型输出名称"""
        return next(iter(self.infer.structured_outputs.keys()))


class OnlineModelFactory(object):
    def __init__(self, model_name, signature_flag="serving_default", logger=None):
        self.model_name = model_name
        self.signature_flag = signature_flag
        self.logger = logger
        self.model = self.create_model()
        self.input_shape = self.model.get_input_shape()
        self.input_tensor_name = self.model.get_input_tensor_name()
        self.output_tensor_name = self.model.get_output_tensor_name()
        self.logger.info(f"模型已经加载，输入尺寸: {self.input_shape}, 输入名称: {self.input_tensor_name}, "
                         f"输出名称: {self.output_tensor_name}")

    def create_model(self):
        model = OnlineModel(self.model_name, self.signature_flag)
        return model

    def infer_data(self, input_data):
        return self.model.infer_data(input_data)




