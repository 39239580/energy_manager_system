import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from core.logging_utils import setup_logging


class LoadData(object):
    def __init__(self, base_line=100, day_points=96, day_nums=365, logger=None):
        self.base_line = base_line
        self.day_points = day_points
        self.day_nums = day_nums
        self.time_points = np.arange(day_nums * day_points)
        self.logger = logger

    def generate_base_load(self):
        """生成更符合实际的合成负荷数据"""
        self.logger.info("模拟负荷数据生成...")
        # 基础负荷模式
        base_load = self.base_line*1.0 + self.base_line*0.50 * np.sin(2 * np.pi * self.time_points / (24 * self.day_points))

        # 周模式
        weekly_pattern = self.base_line*0.20 * np.sin(2 * np.pi * self.time_points / (7 * 24 * self.day_points))

        # 年模式
        seasonal_pattern = self.base_line * 0.30 * np.sin(2 * np.pi * self.time_points / (365 * self.day_points))

        # 工作日/周末模式
        workday_pattern = np.zeros_like(self.time_points)
        for i in range(len(self.time_points)):
            day_of_week = (i // self.day_points) % 7
            if day_of_week < 5:  # 工作日
                workday_pattern[i] = 15 * np.sin(2 * np.pi * (i % self.day_points) / self.day_points)
            else:  # 周末
                workday_pattern[i] = 10 * np.sin(2 * np.pi * (i % self.day_points) / self.day_points)

        # 随机事件和噪声
        random_events = np.zeros_like(self.time_points)
        # 模拟随机事件（例如设备启动）
        event_indices = np.random.choice(len(self.time_points), size=50, replace=False)
        random_events[event_indices] = np.random.uniform(20, 50, size=50)

        noise = 8 * np.random.normal(size=len(self.time_points))

        # 组合所有模式
        load_data = (base_load + weekly_pattern + seasonal_pattern +
                     workday_pattern + random_events + noise)

        # 确保负荷非负
        load_data = np.maximum(load_data, 10)

        return load_data

    @staticmethod
    def add_cyclic_features(df):
        """添加周期性时间特征"""
        # 小时的正弦/余弦表示
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)

        # 分钟特征
        df['minute'] = df['timestamp'].dt.minute

        # 星期的正弦/余弦表示
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)

        # 月份的正弦/余弦表示
        df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)

        # 周末标志
        df['is_weekend'] = df['timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        return df

    @staticmethod
    def add_holiday_features(df, years):
        """添加节假日特征"""
        cn_holidays = holidays.CountryHoliday('CN', years=[years])

        # 节假日当天
        df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in cn_holidays else 0)

        # 节假日前一天
        df['is_pre_holiday'] = df['timestamp'].apply(
            lambda x: 1 if (x + timedelta(days=1)) in cn_holidays else 0
        )

        # 节假日后一天
        df['is_post_holiday'] = df['timestamp'].apply(
            lambda x: 1 if (x - timedelta(days=1)) in cn_holidays else 0
        )

        return df

    def add_real_weather_data(self, df):
        """添加真实天气数据（使用OpenWeatherMap API）"""
        # 实际应用中应调用API，这里简化为模拟
        self.logger.info("模拟天气数据生成...")

        # 更真实的天气模拟
        np.random.seed(42)
        days = len(df) // 96

        # 温度模拟 - 考虑季节性和日变化
        base_temp = 15 + 15 * np.sin(2 * np.pi * np.arange(days) / 365)
        daily_temp_variation = 8 * np.sin(2 * np.pi * np.arange(96) / 96)

        # 添加随机天气波动
        temp_anomaly = np.zeros(days)
        for i in range(0, days, 30):  # 每月一次波动
            if np.random.rand() > 0.7:  # 30%几率有天气事件
                event_duration = np.random.randint(3, 10)
                event_intensity = np.random.uniform(5, 15)
                temp_anomaly[i:i + event_duration] = event_intensity

        # 扩展为每15分钟的数据点
        temp_data = np.zeros(len(df))
        for i in range(days):
            start_idx = i * 96
            end_idx = (i + 1) * 96
            temp_data[start_idx:end_idx] = (
                    base_temp[i] +
                    daily_temp_variation +
                    temp_anomaly[i] +
                    2 * np.random.normal(size=96)
            )

        # 湿度模拟 - 与温度负相关
        humidity = 70 - 0.5 * (temp_data - 20) + 5 * np.random.normal(size=len(df))
        humidity = np.clip(humidity, 30, 95)

        # 添加到DataFrame
        df['temperature'] = temp_data
        df['humidity'] = humidity

        return df

    def prepare_data(self, start_date=(2023, 1, 1)):
        """准备完整数据集"""
        self.logger.info("开始准备数据...")

        # 生成时间序列
        time_points = np.arange(0, self.day_nums * self.day_points)
        start_date = datetime(start_date[0], start_date[1], start_date[2])
        timestamps = [start_date + timedelta(minutes=15 * i) for i in range(len(time_points))]
        # 创建初始DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'load': self.generate_base_load()
        })

        # 添加特征
        df = self.add_cyclic_features(df)
        df = self.add_holiday_features(df, start_date.year)
        df = self.add_real_weather_data(df)
        return df


if __name__ == '__main__':
    logger = setup_logging("fake_load_data.logs")
    LD = LoadData(logger=logger)
    data = LD.prepare_data(start_date=(2023, 1, 1))
    print(data.shape)
    print(data)