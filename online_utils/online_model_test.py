from core.algo.aux_utils import get_quantile_metrics, get_quantile_loss
from online_utils.offline2onlineConvter import ModelConverter
import tensorflow as tf
from offline_utils.data_preprocess_fn import DataProcessUtilsLoder
from online_utils.online_model_utils import OnlineModel

# custom_objects = get_quantile_loss(0.1)
# model_path = "../offline_utils/offline_models/load_forecasting_model"
# converter = ModelConverter(model_path=model_path, custom_objects=custom_objects)
# print("模型加载成功")
# converter.add_signature(tensor_shapes=[(None, 672, 12)], tensor_dtypes=["float32"], tensor_names=["inputs"])
# converter.save_model("./online_models/load_forecasting_model")
# print("模型转换成功")

x = tf.random.uniform(shape=(672, 12), dtype=tf.float32)

# new_model = tf.saved_model.load("./online_models/load_forecasting_model")
# print(new_model.signatures['serving_default'])
# print(new_model.signatures['serving_default'](x))
# print("推断成功")
# y = new_model.signatures['serving_default'](x)["output_0"]
#
# dp = DataProcessUtilsLoder(model_path="../offline_utils/",
#                            data_preprocess_name="test_fake_load_data_preprocess.pkl")
# y_predict = dp.scaler_inverse_transform_y(y, feature_list=12, feature_idx=0)
# print(y_predict)
OM = OnlineModel(model_name="cnn_bi_lstm")
print(OM.infer_data(x))
print(OM.get_input_tensor_name())
print(OM.get_output_tensor_name())
print(OM.get_input_shape())