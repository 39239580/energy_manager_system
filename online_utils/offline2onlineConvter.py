from core.algo.model_tools import KerasModel2TFModel, KerasModel


class ModelConverter(object):  # 模型转换器, 用于生产环境的推断工具
    def __init__(self, model=None, model_path=None, custom_objects=None, compile=True, safe_mode=True, **kwargs):
        self.model = model
        self.model_path = model_path
        self.custom_objects = custom_objects
        self.compile = compile
        self.safe_mode = safe_mode
        if self.model is None:
            if self.model_path is not None:
                self.keras_model = KerasModel.load_model(self.model_path, custom_objects=self.custom_objects,
                                                         compile=self.compile, safe_mode=self.safe_mode, **kwargs)
            else:
                raise ValueError("Either model or model_path should be provided.")
        else:
            self.keras_model = self.model
        self.tf_model = KerasModel2TFModel(self.keras_model)

    def add_signature(self, tensor_shapes, tensor_dtypes, tensor_names=None):
        self.tf_model.set_input_signature(tensor_shapes=tensor_shapes, tensor_dtypes=tensor_dtypes, tensor_names=tensor_names)

    def save_model(self, output_path, signature_flag="serving_default"):
        self.tf_model.save_model(output_path, signature_flag=signature_flag)

