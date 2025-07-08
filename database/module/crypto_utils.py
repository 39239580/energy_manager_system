# 数据加密解密工具类
# crypto_utils.py
import os
import base64
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from database.config import Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CryptoUtils")


class FinancialDataEncryptor:
    def __init__(self, master_key: str, salt_length: int = None, nonce_length: int = None):
        """
        初始化加密器
        :param master_key: 主密钥
        :param salt_length: 盐值长度（可选）
        :param nonce_length: nonce长度（可选）
        """
        self.master_key = master_key.encode('utf-8')
        self.salt_length = salt_length or Config.SALT_LENGTH
        self.nonce_length = nonce_length or Config.NONCE_LENGTH
        self.iterations = Config.PBKDF2_ITERATIONS

    def derive_key(self, salt: bytes) -> bytes:
        """使用PBKDF2从主密钥和盐值派生密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256密钥长度
            salt=salt,
            iterations=self.iterations,
            backend=default_backend()
        )
        return kdf.derive(self.master_key)

    def encrypt(self, plaintext: str) -> dict:
        """
        加密数据
        :param plaintext: 要加密的文本
        :return: 包含加密数据的字典 {'ciphertext', 'salt', 'nonce'}
        """
        # 生成随机盐值
        salt = os.urandom(self.salt_length)
        # 派生密钥
        key = self.derive_key(salt)
        # 生成随机nonce
        nonce = os.urandom(self.nonce_length)

        # 创建加密器
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # 加密数据
        ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()

        return {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'salt': base64.b64encode(salt).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'tag': base64.b64encode(encryptor.tag).decode('utf-8')
        }

    def decrypt(self, encrypted_data: dict) -> str:
        """
        解密数据
        :param encrypted_data: 包含加密数据的字典
        :return: 解密后的文本
        """
        try:
            # 解码base64数据
            ciphertext = base64.b64decode(encrypted_data['ciphertext'])
            salt = base64.b64decode(encrypted_data['salt'])
            nonce = base64.b64decode(encrypted_data['nonce'])
            tag = base64.b64decode(encrypted_data['tag'])

            # 派生密钥
            key = self.derive_key(salt)

            # 创建解密器
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            # 解密数据
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext.decode('utf-8')
        except (KeyError, ValueError, InvalidTag) as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Failed to decrypt data") from e