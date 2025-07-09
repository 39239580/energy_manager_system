import logging
from logging.handlers import RotatingFileHandler


# ======================
# 日志配置
# ======================
def configure_logger():
    # 创建日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器 - 限制单个文件10MB，保留3个备份
    file_handler = RotatingFileHandler(
        'model_service.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=3
    )
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ======================
# 安全配置 - Flask-Talisman
# ======================
csp = {
    'default-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'cdnjs.cloudflare.com'
    ],
    'script-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'cdnjs.cloudflare.com'
    ],
    'style-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'fonts.googleapis.com',
        'cdnjs.cloudflare.com'
    ],
    'font-src': ['\'self\'', 'fonts.gstatic.com', 'data:'],
    'img-src': ['\'self\'', 'data:']
}

MAX_WORKERS = 4  # 根据实际情况调整

class DevelopMODELConfig:
    CNN_BI_LSTM_MODEL = {"MODEL_PATH": "online_models/load_forecasting_model"}
    CNN_LSTM_MODEL = {"MODEL_PATH": "models/cnn_lstm.h5"}
    CNN_GRU_MODEL = {"MODEL_PATH": "models/cnn_gru.h5"}
    CNN_LSTM_ATTENTION_MODEL = {"MODEL_PATH": "models/cnn_lstm_attention.h5"}
    CNN_GRU_ATTENTION_MODEL = {"MODEL_PATH": "models/cnn_gru_attention.h5"}

