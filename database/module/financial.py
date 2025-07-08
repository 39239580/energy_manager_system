from typing import Dict, Optional
from database.base import Database
import json
import logging
from database.module.crypto_utils import FinancialDataEncryptor  # 导入加密工具
from database.config import Config  # 导入配置

logger = logging.getLogger("FinancialModel")


class FinancialAnalysisModel:
    """财务分析管理模块"""
    TABLE_NAME = "financial_analysis"

    # 主密钥 - 在实际应用中应从安全存储获取
    # MASTER_KEY = "your_very_secure_master_key_here"

    def __init__(self, db: Database):
        self.db = db
        # self.encryptor = FinancialDataEncryptor(self.MASTER_KEY)
        self.encryptor = FinancialDataEncryptor(
            Config.FINANCIAL_MASTER_KEY,
            salt_length=Config.SALT_LENGTH,
            nonce_length=Config.NONCE_LENGTH
        )

    def create_table(self):
        """创建财务分析表（带加密支持）"""
        # 先删除旧表（如果存在）
        self.db.drop_table(self.TABLE_NAME)

        # 创建新表结构
        schema = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id": "TEXT NOT NULL",
            "strategy_id": "INTEGER NOT NULL",
            "encrypted_data": "TEXT NOT NULL",  # 加密后的数据
            "salt": "TEXT NOT NULL",  # 盐值
            "nonce": "TEXT NOT NULL",  # nonce
            "tag": "TEXT NOT NULL",  # 认证标签
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "FOREIGN KEY(user_id)": "REFERENCES users(id) ON DELETE CASCADE",
            "FOREIGN KEY(strategy_id)": "REFERENCES strategies(id) ON DELETE CASCADE"
        }
        self.db.create_table(self.TABLE_NAME, schema)
        self.db.create_index(self.TABLE_NAME, "idx_financial_user_strategy",
                             ["user_id", "strategy_id"])
        logger.info("Financial table created with encryption support")

    def drop_table(self):
        self.db.drop_table(self.TABLE_NAME)

    def save_analysis(self, user_id: str, strategy_id: int, financial_data: Dict):
        """保存财务分析结果（加密存储）"""
        # 将财务数据转换为JSON字符串
        plaintext = json.dumps({
            'investment': financial_data['investment'],
            'annual_savings': financial_data['annual_savings'],
            'roi': financial_data['roi'],
            'payback_years': financial_data['payback_years']
        })

        # 加密数据
        encrypted = self.encryptor.encrypt(plaintext)

        # 保存加密后的数据
        self.db.insert(self.TABLE_NAME, {
            "user_id": user_id,
            "strategy_id": strategy_id,
            "encrypted_data": encrypted['ciphertext'],
            "salt": encrypted['salt'],
            "nonce": encrypted['nonce'],
            "tag": encrypted['tag']
        })
        logger.info(f"Encrypted financial analysis saved for strategy: {strategy_id}")

    def load_analysis(self, user_id: str, strategy_id: int) -> Optional[Dict]:
        """加载财务分析结果（解密数据）"""
        result = self.db.select(
            table=self.TABLE_NAME,
            columns=["encrypted_data", "salt", "nonce", "tag", "created_at"],
            condition="user_id = ? AND strategy_id = ?",
            params=(user_id, strategy_id),
            order="ORDER BY created_at DESC",
            limit=1
        )

        if not result:
            return None

        row = result[0]
        encrypted_data = {
            'ciphertext': row["encrypted_data"],
            'salt': row["salt"],
            'nonce': row["nonce"],
            'tag': row["tag"]
        }

        try:
            # 解密数据
            decrypted = self.encryptor.decrypt(encrypted_data)
            # 解析JSON
            financial_data = json.loads(decrypted)
            financial_data['created_at'] = row["created_at"]
            return financial_data
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to decrypt or parse financial data: {str(e)}")
            return None