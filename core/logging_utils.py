import logging


# 配置日志系统
def setup_logging(filename='load_forecast.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setLevel(logging.DEBUG)

    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger