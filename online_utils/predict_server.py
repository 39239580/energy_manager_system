from flask import Flask, request, jsonify
from flask_talisman import Talisman
import concurrent.futures
import numpy as np
import logging
from online_utils.online_model_utils import OnlineModel
import tensorflow as tf


# 配置日志
logging.basicConfig(filename='server.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)

inference_fn = OnlineModel(model_name="cnn_bi_lstm")

# 配置Flask-Talisman安全策略
csp = {
    'default-src': '\'self\'',
    'script-src': '\'self\'',
    'style-src': '\'self\''
}
Talisman(
    app,
    content_security_policy=csp,
    force_https=True,
    frame_options='DENY'
)

# 线程池执行器 - 用于处理并发请求
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def preprocess_data(input_data):
    """将输入数据预处理为模型需要的格式"""
    # 示例：将JSON数组转换为Tensor
    # 根据实际模型输入调整
    return tf.constant([input_data], dtype=tf.float32)


def model_predict(input_tensor):
    """执行模型预测"""
    return inference_fn.infer(input_tensor)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取并验证输入数据
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Invalid input format'}), 400

        # 提交到线程池处理
        future = executor.submit(predict_async, data['input'])
        result = future.result(timeout=10)  # 设置超时时间

        return jsonify({'prediction': result.numpy().tolist()})

    except concurrent.futures.TimeoutError:
        logger.error("Request timed out")
        return jsonify({'error': 'Processing timeout'}), 504
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({'error': str(e)}), 500


def predict_async(input_data):
    """异步处理预测任务"""
    try:
        # 预处理输入
        input_tensor = preprocess_data(input_data)

        # 执行预测
        output = model_predict(input_tensor)

        # 获取第一个输出（根据模型调整）
        return next(iter(output.values()))
    except Exception as e:
        logger.error(f"Async prediction failed: {str(e)}")
        raise


if __name__ == '__main__':
    # 生产环境应使用WSGI服务器如gunicorn
    app.run(host='0.0.0.0', port=5000, threaded=True)

