from core.algo.aux_utils import get_quantile_metrics, get_quantile_loss
from online_utils.offline2onlineConvter import ModelConverter
import tensorflow as tf

# custom_objects = get_quantile_loss(0.1)
# model_path = "../offline_utils/offline_models/load_forecasting_model"
# converter = ModelConverter(model_path=model_path, custom_objects=custom_objects)
# print("模型加载成功")
# converter.add_signature(tensor_shapes=[(None, 672, 12)], tensor_dtypes=["float32"], tensor_names=["inputs"])
# converter.save_model("./online_models/load_forecasting_model")
# print("模型转换成功")

x = tf.random.uniform(shape=(672, 12), dtype=tf.float32)
new_model = tf.saved_model.load("./online_models/load_forecasting_model")
print(new_model.signatures['serving_default'])
print(new_model.signatures['serving_default'](x))
print("推断成功")
print(new_model.signatures['serving_default'](x).numpy())