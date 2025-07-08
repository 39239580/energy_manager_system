from offline_utils.data_preprocess_fn import DataProcessUtils, DataProcessUtilsLoder
from fake_data.load.load_data import LoadData
from core.logging_utils import setup_logging
from core.data_process_utils.process_fn import sk_load

logger = setup_logging("test_fake_load_data.logs")
LD = LoadData(logger=logger)
data = LD.prepare_data(start_date=(2023, 1, 1))
feature_name = ['load', 'hour_sin', 'hour_cos', 'minute', 'day_of_week_sin', 'day_of_week_cos', 'month_sin',
                'month_cos', 'is_weekend', 'is_holiday', 'temperature', 'humidity']
DP = DataProcessUtils(feature_name, test_ratio=0.3)
dataset_ = DP.selector(data)
print(dataset_)
train_dataset, test_dataset = DP.splitter.split_dataset(dataset_)
print(train_dataset)
print(train_dataset.shape)
print("-----------")
train_scaler = DP.scaler.fit_transform(train_dataset)
print(train_scaler)
print(train_scaler.shape)
print(DP.scaler.inverse_transform(train_scaler))

# train_x, train_y, test_x, test_y, _ = DP(data=data)
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
DP.save_scaler("", "test_fake_load_data.pkl")
DP.save_sequence_feature("", "test_fake_load_data_sequence.pkl")
DP.save_data_preprocess("", "test_fake_load_data_preprocess.pkl")
print("保存成功")
print("加载测试")
dp = DataProcessUtilsLoder("", "test_fake_load_data.pkl", "test_fake_load_data_sequence.pkl", "test_fake_load_data_preprocess.pkl",)
# scaler = sk_load("test_fake_load_data.pkl")
# dataset = DP.selector(data)
# print(scaler.transform(dataset))
dataset_ = dp.selector(data)
print(dataset_)
train_dataset_, test_dataset_ = dp.split_dataset(dataset_, test_ratio=0.3, random_state=42)
print(train_dataset_)
print(train_dataset_.shape)
scaler_train_dataset_ = dp.scaler_transform(train_dataset_)
print(scaler_train_dataset_)
print(scaler_train_dataset_.shape)
print(dp.scaler.inverse_transform(scaler_train_dataset_))
# dp.scaler_transform(dataset_)



