import keras
import tensorflow as tf


class ModeInfer(object):  # 模型推断工具 跨平台使用, 保存重新加载的模型，不具有predict 和 evaluate 方法
    def __init__(self, model=None, model_path=None):
        self.model = model
        self.model_path = model_path
        if self.model is None:  # 模型对象为空时，尝试加载模型
            if self.model_path is None:
                self.model = None
                self.infer = None
            else:
                self.model = self._load_model()
                self.infer = self._init_infer()  # 加载推断期器
        else:
            self.infer = self._init_infer()

    @classmethod
    def load_model(cls, model_path=None):
        return tf.saved_model.load(cls, model_path)

    def _load_model(self):
        return tf.saved_model.load(self.model_path)

    def _init_infer(self, signature_flag="serving_default"):  # 初始化推断器件
        return self.model.signatures[signature_flag]

    @classmethod
    def init_infer(cls, model, signature_flag="serving_default"):  # 初始化推断器件
        return model.signatures[signature_flag]

    def infer(self, input_data):  # 推断数据
        return self.infer(input_data)

    @classmethod
    def infer_(cls, infer, input_data):  # 推断数据
        return infer(input_data)

    @property
    def get_model(self):
        return self.model


class ModelInSignature(object):  # 模型签名工具 跨平台使用
    def __init__(self):
        self.dtypes = self._data_type()

    @staticmethod
    def _data_type():
        return {"float32": tf.float32, "float64": tf.float64, "float16": tf.float16,
                "int8": tf.int8, "int16": tf.int16, "int32": tf.int32, "int64": tf.int64,
                "string": tf.string}

    def set_tensor_spec(self, tensor_shape, tensor_dtype="float32", tensor_name=None):

        return tf.TensorSpec(shape=tensor_shape, dtype=self.dtypes[tensor_dtype], name=tensor_name)

    def set_multiple_tensor_spec(self, tensor_shapes, tensor_dtypes, tensor_names=None):
        tensor_specs = []
        for i in range(len(tensor_shapes)):
            tensor_specs.append(self.set_tensor_spec(tensor_shapes[i], tensor_dtypes[i], tensor_names[i]))
        return tensor_specs

    def set_multiple_tensor_spec_dict(self, tensor_shapes_dict, tensor_dtypes_dict, tensor_names_dict=None):
        tensor_specs_dict = {}
        for key in tensor_shapes_dict:
            tensor_specs_dict[key] = self.set_multiple_tensor_spec(tensor_shapes_dict[key], tensor_dtypes_dict[key],
                                                                   tensor_names_dict[key])
        return tensor_specs_dict


class KerasModel2TFModel(object):  # keras模型转tf模型, 可被继承
    def __init__(self, model):
        self.model = model
        self.signature = ModelInSignature()
        self.server = None

    def set_input_signature(self, tensor_shapes, tensor_dtypes, tensor_names=None):
        if isinstance(tensor_shapes, list):
            tensor_specs = self.signature.set_multiple_tensor_spec(tensor_shapes, tensor_dtypes, tensor_names)
        elif isinstance(tensor_shapes, dict):
            tensor_specs = self.signature.set_multiple_tensor_spec_dict(tensor_shapes, tensor_dtypes, tensor_names)
        else:
            raise ValueError("tensor_shapes must be list or dict")
        self.server = tensor_specs
        return tensor_specs

    def exec_signature(self, signature_flag="serving_default"):
        if isinstance(self.server, dict):
            @tf.function(input_signature=[self.server])
            def serve(inputs):
                return self.model(inputs)

        else:
            @tf.function(input_signature=self.server)
            def serve(*inputs):
                if len(self.server) == 1:
                    return self.model(inputs[0])
                return self.model(inputs)
        return {signature_flag: serve}

    def save_model(self, model_path, signature_flag="serving_default"):  # 保存tf模型, 默认只有一个serving_default签名
        tf.saved_model.save(self.model, model_path, signatures=self.exec_signature(signature_flag))


class KerasModel(object):  # 单纯的kerasModel 的保存与加载与预测和评估 不用于生产的推断工具
    def __init__(self, model=None, model_path=None, custom_objects=None, compile=True, safe_mode=True, **kwargs):
        """
        :param model:
        :param model_path:
        :param custom_objects:
        :param compile:
        :param safe_mode:
        :param kwargs:
        """
        self.model = model
        self.model_path = model_path
        self.custom_objects = custom_objects
        self.compile = compile
        self.safe_mode = safe_mode
        self.kwargs = kwargs
        if self.model is None:
            if self.model_path is None:
                self.model = None
            else:
                self.model = self._load_model()

    @classmethod
    def load_model(cls, model_path=None, custom_objects=None, compile=True, safe_mode=True, **kwargs):
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=compile,
                                          safe_mode=safe_mode, **kwargs)

    @classmethod
    def predict_(cls, model, input_data):
        return model.predict(input_data)

    def _load_model(self):
        return tf.keras.models.load_model(self.model_path, custom_objects=self.custom_objects, compile=self.compile,
                                          safe_mode=self.safe_mode, **self.kwargs)

    def predict(self, input_data, model_path=None):
        if model_path is not None:
            self.model_path = model_path
            self.model = self._load_model()
        return self.model.predict(input_data)

    def evaluate(self, x=None, y=None, batch_size=None, model_path=None, **kwargs):
        if model_path is not None:
            self.model_path = model_path
            self.model = self._load_model()
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, **kwargs)

    @classmethod
    def evaluate_(cls, model, x=None, y=None, batch_size=None, **kwargs):
        return model.evaluate(x=x, y=y, batch_size=batch_size, **kwargs)

    @property
    def get_model(self):
        return self.model

    def save_model(self, model, filepath, overwrite=True, save_format=None, **kwargs):
        tf.keras.models.save_model(self.model, model, filepath, overwrite, save_format, **kwargs)


class ComplexModel(keras.Model):  # 负载模型
    def __init__(self, sub_model1, sub_model2, final_units=1, merge_type='concat', use_concat=True, **kwargs):
        super(ComplexModel, self).__init__(**kwargs)
        self.sub_model1 = sub_model1  # 模型1
        self.sub_model2 = sub_model2  # 模型2
        self.final_units = final_units  # 输出维度
        self.merge_type = merge_type  # 合并方式
        self.use_concat = use_concat  # 是否使用concat合并
        if self.merge_type == 'concat':
            self.merge_layer = self._merge_concat()  # 合并层
        else:
            self.merge_layer = self._merge_add()  # 合并层
        self.final_layer = self._final_layer()

    @staticmethod
    def _merge_concat():
        return keras.layers.Concatenate()

    @staticmethod
    def _merge_add():  #
        return keras.layers.Add()

    def _final_layer(self):
        if self.final_units == 1:
            return keras.layers.Dense(self.final_units, activation='sigmoid')
        else:
            return keras.layers.Dense(self.final_units, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        if self.use_concat:
            x1 = self.sub_model1(inputs, training=training, mask=mask)
            x2 = self.sub_model2(inputs, training=training, mask=mask)
            x = self.merge_layer([x1, x2])
        else:
            x = self.sub_model1(inputs, training=training, mask=mask)
        final_output = self.final_layer(x)
        return final_output

    def get_config(self):
        config = super(ComplexModel, self).get_config()
        config.update({
            'sub_model1': self.sub_model1,
            'sub_model2': self.sub_model2,
            'final_units': self.final_units,
            'merge_type': self.merge_type,
            'use_concat': self.use_concat
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TFLiteModel(object):  # 用于生产的tflite模型
    def __init__(self, model, saved_model_dir, input_shapes, output_shapes, quantize=False, signature_keys=None,
                 tags=None, inference_input_type=tf.float16, inference_output_type=tf.float16, **kwargs):
        self.model = model
        self.saved_model_dir = saved_model_dir
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.quantize = quantize
        self.signature_keys = signature_keys
        self.tags = tags
        self.inference_input_type = inference_input_type
        self.inference_output_type = inference_output_type
        self.kwargs = kwargs
        if self.model is None:
            self._from_keras_model(self.model)
        else:
            self._from_saved_model(self.saved_model_dir, self.signature_keys, self.tags)
        if self.quantize:
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                        tf.lite.OpsSet.SELECT_TF_OPS]
            self.converter.inference_input_type = self.inference_input_type
            self.converter.inference_output_type = self.inference_output_type
        self.tflite_model = self.converter.convert()

    def _from_keras_model(self, model):
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)

    def _from_saved_model(self, saved_model_dir, signature_keys=None, tags=None):
        self.converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=signature_keys,
                                                                  tags=tags)

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            f.write(self.tflite_model)

    @classmethod
    def load_model(cls, model_path, **kwargs):
        with open(model_path, 'rb') as f:
            tflite_model = f.read()
        return tf.lite.Interpreter(model_content=tflite_model, **kwargs)

    def predict(self, input_data, interpreter=None):
        if interpreter is None:
            interpreter = tf.lite.Interpreter(model_content=self.tflite_model, **self.kwargs)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        for i in range(len(input_data)):
            interpreter.set_tensor(input_details[i]['index'], input_data[i])
        interpreter.invoke()  # 执行推断
        output_data = []
        for i in range(len(output_details)):
            output_data.append(interpreter.get_tensor(output_details[i]['index']))
        return output_data

