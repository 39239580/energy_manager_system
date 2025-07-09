from flask import Flask, request, jsonify
import numpy as np
from online_utils.online_model_utils import OnlineModel, OnlineModelFactory
from flask import Flask, request, jsonify
import json
from concurrent.futures import ThreadPoolExecutor
from flask_talisman import Talisman
import tensorflow as tf
from online_utils.model_config import configure_logger, csp
from online_utils.model_config import MAX_WORKERS
import atexit
import concurrent.futures

app = Flask(__name__)
logger = configure_logger()
Talisman(
    app,
    content_security_policy=csp,
    content_security_policy_nonce_in=['script-src', 'style-src'],
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    frame_options='DENY'
)


# ======================
# 模型加载
# ======================
logger.info("正在加载TensorFlow模型...")
try:
    # 加载SavedModel
    OM = OnlineModelFactory(model_name="cnn_bi_lstm", signature_flag="serving_default", logger=logger)
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise RuntimeError("无法加载模型") from e

# ======================
# 根据服务器CPU核心数设置线程池大小
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# ======================
# 优雅关闭
# ======================
def shutdown_handler():
    logger.info("关闭线程池执行器...")
    executor.shutdown(wait=True)
    logger.info("服务已安全关闭")


atexit.register(shutdown_handler)


def preprocess_data(input_data):
    """更健壮的预处理函数"""
    # 转换为numpy数组
    try:
        array_data = np.array(input_data, dtype=np.float32)
    except ValueError as e:
        raise ValueError(f"Invalid input data format: {str(e)}")

    # 自动添加批次维度 (如果输入是单样本)
    if array_data.ndim == len(OM.input_shape):
        array_data = np.expand_dims(array_data, axis=0)

    # 验证输入形状
    expected_shape = OM.input_shape
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

    return tf.constant(array_data)


@app.route('/predict', methods=['POST'])
def predict():
    # 获取客户端IP用于日志
    client_ip = request.remote_addr
    extra = {'client_ip': client_ip}
    logger.info(f"Received prediction request", extra=extra)

    try:
        data = request.get_json()
        if not data or 'input' not in data:
            logger.warning("Invalid request: missing 'input' field", extra=extra)
            return jsonify({'error': 'Missing input data'}), 400

        # 提交到线程池处理
        future = executor.submit(predict_async, data['input'], client_ip)
        result = future.result(timeout=10)  # 设置超时时间

        # 转换为Python原生类型
        if isinstance(result, tf.Tensor):
            result = result.numpy().tolist()

        logger.info("Prediction successful", extra=extra)
        return jsonify({'prediction': result})

    except concurrent.futures.TimeoutError:
        logger.error("Request processing timed out", extra=extra)
        return jsonify({'error': 'Processing timeout'}), 504
    except ValueError as ve:
        logger.error(f"Input validation failed: {str(ve)}", extra=extra)
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.exception("Unexpected error during prediction", extra=extra)
        return jsonify({'error': 'Internal server error'}), 500


def predict_async(input_data, client_ip):
    """异步处理预测任务"""
    extra = {'client_ip': client_ip}

    try:
        logger.debug("Starting input preprocessing", extra=extra)
        input_tensor = preprocess_data(input_data)

        logger.debug(f"Input tensor shape: {input_tensor.shape}", extra=extra)

        # 执行预测
        output = OM.infer_data(input_tensor)

        # 获取输出结果
        output_tensor = output[OM.output_tensor_name]

        logger.debug(f"Output tensor shape: {output_tensor.shape}", extra=extra)
        return output_tensor

    except Exception as e:
        logger.error(f"Async prediction failed: {str(e)}", extra=extra)
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'thread_pool': {
            'max_workers': executor._max_workers,
            'pending_tasks': executor._work_queue.qsize(),
            'active_threads': len(executor._threads)
        }
    })


if __name__ == '__main__':
    # 生产环境应使用WSGI服务器如gunicorn
    logger.info("Starting Flask server")
    # app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)

