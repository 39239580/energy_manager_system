import keras
import tfx_bsl.coders.tf_graph_record_decoder

from core.components.deep_pipeline import DeepPipeline
from fake_data.load.load_data import LoadData
from core.logging_utils import setup_logging
from core.algo.callback_utils import DeepModelCallback
from core.algo.aux_utils import get_quantile_metrics, get_quantile_loss
from core.algo.model_tools import KerasModel, KerasModel2TFModel
import tensorflow as tf
# from core.algo.model_tools import KerasModel2TFModel
from core.algo.aux_utils import quantile_loss


logger = setup_logging("offline_logs/train_config.logs")
LD = LoadData(logger=logger)
data = LD.prepare_data(start_date=(2023, 1, 1))
feature_name = ['load', 'hour_sin', 'hour_cos', 'minute', 'day_of_week_sin', 'day_of_week_cos', 'month_sin',
                'month_cos', 'is_weekend', 'is_holiday', 'temperature', 'humidity']

DP = DeepPipeline(feature_name, target_name="load", target_index=0, window_size=96 * 7, step_size=96)
train_x, train_y, test_x, test_y, _ = DP.process_fit_transform(data)
print(train_x.shape)
print(train_y.shape)
train_x, val_x, train_y, val_y = DP.process_validation(x=train_x, y=train_y)
optimizer = 'adam'
loss = 'mse'
DP.compile_model(optimizer="adam", loss="mse", quantile_q=0.1)
DP.model.build(input_shape=(None, 672, 12))
DP.summary_model()
logger.info("模型训练...")
DP.fit_model(train_x, train_y, val_x, val_y, batch_size=64, epochs=1, verbose=1, workers=4, use_multiprocessing=True)
logger.info("模型保存...")
DP.save_model("offline_models/load_forecasting_model", save_format="tf")

# # logger.info("加载模型...")
# custom_objects = get_quantile_loss(0.1)
# model = KerasModel.load_model("offline_models/load_forecasting_model", custom_objects=custom_objects)
# # logger.info("模型评估...")
# # eval_results = model.evaluate(test_x, test_y, batch_size=64, verbose=1)
# # print(eval_results)
# logger.info("模型预测...")
# predict_result = model.predict(test_x)
# print(predict_result)
# print(predict_result.shape)
# logger.info("模型save_model 保存...")
# KM2TM = KerasModel2TFModel(model)
# KM2TM.set_input_signature(tensor_shapes=[(None, 672, 12)], tensor_dtypes=["float32"], tensor_names=['inputs'])
# KM2TM.save_model("offline_models/load_forecasting_model_saved_model", signature_flag="serving_default")
# logger.info("save_model模型 加载...")
# new_model = tf.saved_model.load("offline_models/load_forecasting_model_saved_model")
# print(new_model.signatures["serving_default"])
# x = tf.random.uniform(shape=(672, 12), dtype=tf.float32)
# print(new_model.signatures['serving_default'](x)["output_0"])
# # eval_results = new_model.evaluate(test_x, test_y, batch_size=64, verbose=1)
# # print(eval_results)
# # model2 = keras.Model()
# # model2.evaluate



