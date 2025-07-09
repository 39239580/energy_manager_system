from flask_talisman import Talisman
from online_utils.online_model_utils import OnlineModel, OnlineModelFactory
from concurrent.futures import ThreadPoolExecutor
from online_utils.model_config import MAX_WORKERS
import atexit
import numpy as np
import tensorflow as tf
import requests
import csv
from requests.exceptions import RequestException


def talisman_wrapper(app, csp):
    Talisman(
        app,
        content_security_policy=csp,
        content_security_policy_nonce_in=['script-src', 'style-src'],
        force_https=True,
        strict_transport_security=True,
        session_cookie_secure=True,
        frame_options='DENY'
    )


def load_model(logger):
    logger.info("正在加载TensorFlow模型...")
    try:
        # 加载SavedModel
        OM = OnlineModelFactory(model_name="cnn_bi_lstm", signature_flag="serving_default", logger=logger)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise RuntimeError("无法加载模型") from e
    return OM


def set_thread_pool_executor(logger):
    # ======================
    # 根据服务器CPU核心数设置线程池大小
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # ======================
    # 优雅关闭
    def shutdown_handler():
        logger.info("关闭线程池执行器...")
        executor.shutdown(wait=True)
        logger.info("服务已安全关闭")

    atexit.register(shutdown_handler)
    return executor


def preprocess_data(input_data, input_shape):
    """更健壮的预处理函数"""
    # 转换为numpy数组
    try:
        array_data = np.array(input_data, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Invalid input data format: {str(e)}")

    # 自动添加批次维度 (如果输入是单样本)
    if array_data.ndim == len(input_shape):
        array_data = np.expand_dims(array_data, axis=0)

    # 验证输入形状
    expected_shape = input_shape
    actual_shape = array_data.shape[1:]

    # 处理动态维度（None值）
    if len(expected_shape) != len(actual_shape):
        raise ValueError(
            f"Input rank mismatch. Expected {len(expected_shape)} dimensions, got {len(actual_shape)}"
        )

    for i, (exp_dim, act_dim) in enumerate(zip(expected_shape, actual_shape)):
        if exp_dim is not None and exp_dim != act_dim:
            raise ValueError(
                f"Dimension {i} mismatch. Expected {exp_dim}, got {act_dim}"
            )
    return tf.convert_to_tensor(array_data)
    # return tf.constant(array_data)


def fetch_external_data(source_config):
    """从外部数据源获取特征数据"""
    try:
        source_type = source_config.get('type')

        if source_type == 'api':
            # 从API获取数据
            url = source_config['url']
            params = source_config.get('params', {})
            headers = source_config.get('headers', {})

            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()

            data = response.json()
            # 提取特征路径
            feature_path = source_config.get('feature_path', 'data')
            features = data
            for key in feature_path.split('.'):
                features = features[key]

            return np.array(features, dtype=np.float32)

        elif source_type == 'database':  # 从数据库获取数据
            # 模拟数据库查询
            # 实际实现需要根据具体数据库调整
            raise NotImplementedError("Database integration not implemented")

        elif source_type == 'file':  # 从文件系统获取数据
            # 从文件系统获取数据
            file_path = source_config['path']
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
            # 跳过标题行（如果有）
            if source_config.get('has_header', False):
                data = data[1:]
            return np.array(data, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported data source type: {source_type}")

    except RequestException as e:
        raise ConnectionError(f"Failed to fetch data from external source: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error processing external data: {str(e)}")