import uuid
import bcrypt
import logging
from typing import Tuple, Optional
from database.base import Database

logger = logging.getLogger("UserModel")


class UserModel(object):
    """用户管理模块"""
    TABLE_NAME = "users"

    def __init__(self, db: Database):
        self.db = db

    def create_table(self):
        schema = {
            "id": "TEXT PRIMARY KEY",
            "username": "TEXT UNIQUE NOT NULL",
            "password": "TEXT NOT NULL",
            "company_name": "TEXT NOT NULL",
            "industry_type": "TEXT NOT NULL",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        self.db.create_table(self.TABLE_NAME, schema)
        self.db.create_index(self.TABLE_NAME, "idx_users_username", ["username"])

    def drop_table(self):
        self.db.drop_table(self.TABLE_NAME)

    def add_user(self, username: str, password: str, company_name: str, industry_type: str) -> str:
        """添加新用户"""
        if not username or not password:
            raise ValueError("Username and password are required")

        user_id = str(uuid.uuid4())
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        user_data = {
            "id": user_id,
            "username": username,
            "password": hashed_password,
            "company_name": company_name,
            "industry_type": industry_type
        }

        self.db.insert(self.TABLE_NAME, user_data)
        logger.info(f"User added: {username}")
        return user_id

    def authenticate_user(self, username: str, password: str) -> Optional[Tuple]:
        """验证用户"""
        result = self.db.select(
            table=self.TABLE_NAME,
            columns=["id", "password", "company_name", "industry_type"],
            condition="username = ?",
            params=(username,)
        )

        if not result:
            return None

        user_data = result[0]
        user_id = user_data["id"]
        stored_hash = user_data["password"]
        company_name = user_data["company_name"]
        industry_type = user_data["industry_type"]

        if bcrypt.checkpw(password.encode(), stored_hash.encode()):
            return (user_id, company_name, industry_type)
        return None

    def delete_user(self, user_id: str):
        """删除用户及其所有关联数据（不开启新事务）"""
        # 直接执行删除操作，不开启新事务
        self.db.delete(self.TABLE_NAME, "id = ?", (user_id,))
        logger.info(f"User deleted: {user_id}")