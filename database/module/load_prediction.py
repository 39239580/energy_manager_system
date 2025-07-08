import pandas as pd
import logging
from database.base import Database
from typing import Optional


logger = logging.getLogger("LoadPredictionModel")


class LoadPredictionModel(object):
    """负荷预测结果管理模块"""
    TABLE_NAME = "load_predictions"

    def __init__(self, db: Database):
        self.db = db

    def create_table(self):
        schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id": "TEXT NOT NULL",
            "prediction_type": "TEXT NOT NULL",
            "timestamp": "TIMESTAMP NOT NULL",
            "value": "REAL NOT NULL",
            "FOREIGN KEY(user_id)": "REFERENCES users(id) ON DELETE CASCADE"
        }
        self.db.create_table(self.TABLE_NAME, schema)
        self.db.create_index(self.TABLE_NAME, "idx_preds_user_type_time",
                             ["user_id", "prediction_type", "timestamp"])

    def drop_table(self):
        self.db.drop_table(self.TABLE_NAME)

    def save_prediction(self, user_id: str, prediction_type: str, df: pd.DataFrame):
        """保存预测结果（修复数据类型问题）"""
        data = []
        for _, row in df.iterrows():
            # 将 Pandas 类型转换为基本 Python 类型
            timestamp = row['timestamp']
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()

            data.append((
                user_id,
                prediction_type,
                timestamp,
                float(row['value'])
            ))

        columns = ["user_id", "prediction_type", "timestamp", "value"]

        # 执行操作
        self.db.delete(
            self.TABLE_NAME,
            "user_id = ? AND prediction_type = ?",
            (user_id, prediction_type)
        )
        self.db.bulk_insert(self.TABLE_NAME, columns, data)

        logger.info(f"Prediction saved: {prediction_type} for user: {user_id}")

    def load_predictions(self, user_id: str, prediction_type: str) -> Optional[pd.DataFrame]:
        """加载预测结果"""
        result = self.db.select(
            table=self.TABLE_NAME,
            columns=["timestamp", "value"],
            condition="user_id = ? AND prediction_type = ?",
            params=(user_id, prediction_type)
        )

        if not result:
            return None

        data = [dict(row) for row in result]
        return pd.DataFrame(data).rename(columns={"value": prediction_type})