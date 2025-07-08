# config.py
import os


class Config:
    # 从环境变量获取主密钥，如果不存在则使用默认值（仅用于开发）
    FINANCIAL_MASTER_KEY = os.getenv("FINANCIAL_MASTER_KEY", "default_insecure_key_do_not_use_in_prod")

    # 加密参数
    PBKDF2_ITERATIONS = 100000
    SALT_LENGTH = 16
    NONCE_LENGTH = 12