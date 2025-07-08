from fake_data.load.load_data import LoadData
from core.logging_utils import setup_logging
from core.data_process_utils.process_fn import FeatureSelector
from core.data_process_utils.dataset_split_utils import DatasetSplitUtils
from core.data_process_utils.process_fn import Scaler
from core.data_process_utils.process_fn import BuildSequenceFeature

logger = setup_logging("fake_load_data.logs")
LD = LoadData(logger=logger)
data = LD.prepare_data(start_date=(2023, 1, 1))
print(data.shape)
print(data)
feature_name = ['load', 'hour_sin', 'hour_cos', 'minute', 'day_of_week_sin', 'day_of_week_cos', 'month_sin',
                'month_cos', 'is_weekend', 'is_holiday', 'temperature', 'humidity']
FS = FeatureSelector(feature_name=feature_name, return_type=None)
df, df_array = FS(data)
logger.info("数据集划分...")
DS = DatasetSplitUtils(test_ratio=0.3)
train_data, test_data = DS.split_dataset(df_array)
print(train_data.shape)
print(test_data.shape)
train_df, test_df = DS.split_dataset(df)
print(train_df)
print(train_df.shape)
print(test_df.shape)
print("train_data")
print(train_df)
logger.info("数据标准化...")
S = Scaler(scaler_type='min-max')
train_scaler = S.fit_transform(train_data)
print(train_scaler)
train_scaler_df = S.fit_transform(train_df)
print(train_scaler_df)
test_scaler = S.transform(test_data)
test_scaler_df = S.transform(test_df)
print(test_scaler)
print(test_scaler_df)
S.save("scaler.pkl")

scaler_model = Scaler.load("scaler.pkl")
print("但反归一，恢复数据， train_data")
print(scaler_model.inverse_transform(train_scaler))
print("数据构造测试转成数组")
logger.info("序列特征构造...")
BSF = BuildSequenceFeature(feature_name=feature_name[1:])
x_train, y_train = BSF(train_scaler)
print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)
x_test, y_test = BSF(train_scaler_df)
print(x_test)
print(x_test.shape)
print(y_test)
print(y_test.shape)

S = Scaler(scaler_type='min-max')
L = S.fit(train_data)
print(L)
print(L.transform(train_data))
S.save("scaler_.pkl")
scaler_model = Scaler.load("scaler_.pkl")
print(scaler_model.transform(train_data))

