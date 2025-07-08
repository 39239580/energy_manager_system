import pandas as pd
import logging
from database.base import Database
from typing import Optional

logger = logging.getLogger("DataModel")


class HistoricalDataModel(object):
    """历史数据管理模块"""
    TABLE_NAME = "historical_data"

    def __init__(self, db: Database):
        self.db = db

    def create_table(self):
        schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id": "TEXT NOT NULL",
            "timestamp": "TIMESTAMP NOT NULL",
            "load": "REAL NOT NULL",
            "temperature": "REAL NOT NULL",
            "humidity": "REAL NOT NULL",
            "electricity_price": "REAL NOT NULL",
            "FOREIGN KEY(user_id)": "REFERENCES users(id) ON DELETE CASCADE"
        }
        self.db.create_table(self.TABLE_NAME, schema)
        self.db.create_index(self.TABLE_NAME, "idx_hist_data_user_time", ["user_id", "timestamp"])

    def drop_table(self):
        self.db.drop_table(self.TABLE_NAME)

    def save_data(self, user_id: str, df: pd.DataFrame):
        """保存历史数据（修复数据类型问题）"""
        data = []
        for _, row in df.iterrows():
            # 将 Pandas 类型转换为基本 Python 类型
            timestamp = row['timestamp']
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()

            data.append((
                user_id,
                timestamp,
                float(row['load']),
                float(row['temperature']),
                float(row['humidity']),
                float(row['electricity_price'])
            ))

        columns = ["user_id", "timestamp", "load", "temperature", "humidity", "electricity_price"]

        # 执行操作
        self.db.delete(self.TABLE_NAME, "user_id = ?", (user_id,))
        self.db.bulk_insert(self.TABLE_NAME, columns, data)

        logger.info(f"Historical data saved for user: {user_id}")

    def load_data(self, user_id: str) -> Optional[pd.DataFrame]:
        """加载历史数据"""
        result = self.db.select(
            table=self.TABLE_NAME,
            columns=["timestamp", "load", "temperature", "humidity", "electricity_price"],
            condition="user_id = ?",
            params=(user_id,)
        )

        if not result:
            return None

        data = [dict(row) for row in result]
        return pd.DataFrame(data)