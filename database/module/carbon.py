import logging
from typing import Optional
from database.base import Database

logger = logging.getLogger("CarbonModel")


class CarbonManagementModel:
    """碳管理平台相关模块"""
    CONFIG_TABLE = "carbon_config"
    ACTIVITY_TABLE = "carbon_activities"
    EMISSION_TABLE = "carbon_emissions"

    def __init__(self, db: Database):
        self.db = db

    def create_table(self):
        # 碳平台配置表
        config_schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "config_key": "TEXT UNIQUE NOT NULL",
            "config_value": "TEXT NOT NULL",
            "description": "TEXT",
            "last_updated": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        self.db.create_table(self.CONFIG_TABLE, config_schema)
        self.db.create_index(self.CONFIG_TABLE, "idx_carbon_config_key", ["config_key"])

        # 碳平台活动表
        activity_schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id": "TEXT NOT NULL",
            "activity_type": "TEXT NOT NULL",
            "activity_time": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "details": "TEXT",
            "FOREIGN KEY(user_id)": "REFERENCES users(id) ON DELETE CASCADE"
        }
        self.db.create_table(self.ACTIVITY_TABLE, activity_schema)
        self.db.create_index(self.ACTIVITY_TABLE, "idx_carbon_activities_user", ["user_id"])

        # 碳排放数据表
        emission_schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id": "TEXT NOT NULL",
            "timestamp": "TIMESTAMP NOT NULL",
            "scope1": "REAL NOT NULL",
            "scope2": "REAL NOT NULL",
            "scope3": "REAL NOT NULL",
            "total": "REAL NOT NULL",
            "FOREIGN KEY(user_id)": "REFERENCES users(id) ON DELETE CASCADE"
        }
        self.db.create_table(self.EMISSION_TABLE, emission_schema)
        self.db.create_index(self.EMISSION_TABLE, "idx_emissions_user_time", ["user_id", "timestamp"])

    def drop_table(self):
        self.db.drop_table(self.CONFIG_TABLE)
        self.db.drop_table(self.ACTIVITY_TABLE)
        self.db.drop_table(self.EMISSION_TABLE)

    def save_config(self, config_key: str, config_value: str, description: str = ""):
        """保存碳平台配置"""
        self.db.execute(
            f"INSERT OR REPLACE INTO {self.CONFIG_TABLE} "
            "(config_key, config_value, description) "
            "VALUES (?, ?, ?)",
            (config_key, config_value, description)
        )
        logger.info(f"Carbon config updated: {config_key}")

    def get_config(self, config_key: str) -> Optional[str]:
        """获取碳平台配置"""
        result = self.db.select(
            table=self.CONFIG_TABLE,
            columns=["config_value"],
            condition="config_key = ?",
            params=(config_key,)
        )
        return result[0]["config_value"] if result else None

    def log_activity(self, user_id: str, activity_type: str, details: str = ""):
        """记录碳平台用户活动"""
        self.db.insert(self.ACTIVITY_TABLE, {
            "user_id": user_id,
            "activity_type": activity_type,
            "details": details
        })
        logger.info(f"Carbon activity logged: {activity_type} for user: {user_id}")

    def save_emission_data(self, user_id: str, emission_data: dict):
        """保存碳排放数据（修复数据类型问题）"""
        # 确保所有值为基本Python类型
        converted_data = {
            "user_id": user_id,
            "timestamp": emission_data['timestamp'],
            "scope1": float(emission_data['scope1']),
            "scope2": float(emission_data['scope2']),
            "scope3": float(emission_data['scope3']),
            "total": float(emission_data['total'])
        }

        self.db.insert(self.EMISSION_TABLE, converted_data)
        logger.info(f"Carbon emission data saved for user: {user_id}")

    def load_emission_data(self, user_id: str) -> Optional[dict]:
        """加载碳排放数据"""
        result = self.db.select(
            table=self.EMISSION_TABLE,
            columns=["timestamp", "scope1", "scope2", "scope3", "total"],
            condition="user_id = ?",
            params=(user_id,)
        )
        return [dict(row) for row in result] if result else None