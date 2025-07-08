import json
import logging
from typing import List, Dict
from database.base import Database

logger = logging.getLogger("StrategyModel")


class StrategyModel(object):
    """策略配置管理模块"""
    TABLE_NAME = "strategies"

    def __init__(self, db: Database):
        self.db = db

    def create_table(self):
        schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id": "TEXT NOT NULL",
            "strategy_name": "TEXT NOT NULL",
            "parameters": "TEXT NOT NULL",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "FOREIGN KEY(user_id)": "REFERENCES users(id) ON DELETE CASCADE"
        }
        self.db.create_table(self.TABLE_NAME, schema)
        self.db.create_index(self.TABLE_NAME, "idx_strategies_user", ["user_id"])

    def drop_table(self):
        self.db.drop_table(self.TABLE_NAME)

    def save_strategy(self, user_id: str, strategy_name: str, parameters: Dict) -> int:
        """保存策略配置"""
        params_json = json.dumps(parameters)
        strategy_id = self.db.insert(self.TABLE_NAME, {
            "user_id": user_id,
            "strategy_name": strategy_name,
            "parameters": params_json
        })
        logger.info(f"Strategy saved: {strategy_name} for user: {user_id}")
        return strategy_id

    def load_strategies(self, user_id: str) -> List[Dict]:
        """加载用户的所有策略"""
        result = self.db.select(
            table=self.TABLE_NAME,
            columns=["id", "strategy_name", "parameters", "created_at"],
            condition="user_id = ?",
            params=(user_id,)
        )

        strategies = []
        for row in result:
            strategies.append({
                'id': row["id"],
                'name': row["strategy_name"],
                'parameters': json.loads(row["parameters"]),
                'created_at': row["created_at"]
            })

        return strategies