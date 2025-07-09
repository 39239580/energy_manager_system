import numpy as np

from online_utils.server_utils import talisman_wrapper, load_model, set_thread_pool_executor, preprocess_data, \
    fetch_external_data
from flask import Flask, request, jsonify, send_file
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from online_utils.model_config import configure_logger, csp
import concurrent.futures
import requests
from requests.exceptions import RequestException
import csv
import os
import io
import uuid
import tempfile


app = Flask(__name__)
logger = configure_logger()
talisman_wrapper(app, csp)
# ======================
# 模型加载
OM = load_model(logger=logger)
# 设置线程池执行器
executor = set_thread_pool_executor(logger=logger)


@app.route('/predict', methods=['POST'])
def predict():
    """处理JSON格式的预测请求"""
    # 获取客户端IP用于日志
    client_ip = request.remote_addr  # 客户端ip
    extra = {'client_ip': client_ip}
    logger.info(f"Received prediction request", extra=extra)

    try:
        data = request.get_json()  # 获取到的JSON数据
        if not data:  # 数据为空
            logger.warning("Invalid request: empty JSON body", extra=extra)
            return jsonify({'error': 'Missing input data'}), 400

        # 处理多源数据输入
        if 'sources' in data:
            # 多源数据模式
            all_features = []
            for source in data['sources']:
                source_type = source.get('type', 'direct')

                if source_type == 'direct':
                    # 直接提供特征数据
                    if 'features' not in source:
                        raise ValueError("Missing 'features' in direct source")
                    features = np.array(source['features'], dtype=np.float32)  #类似一个列表，转成数组数据
                    all_features.append(features)

                elif source_type == 'external':  # 来自外部数据
                    # 从外部数据源获取
                    if 'config' not in source:
                        raise ValueError("Missing 'config' for external source")
                    features = fetch_external_data(source['config'])
                    all_features.append(features)

                else:
                    raise ValueError(f"Unsupported source type: {source_type}")

            # 合并所有特征
            if len(all_features) == 1:
                input_data = all_features[0]
            else:
                # 水平拼接特征
                input_data = np.hstack(all_features)
        else:
            # 单源数据模式
            if 'input' not in data:
                logger.warning("Invalid request: missing 'input' field", extra=extra)
                return jsonify({'error': 'Missing input data'}), 400
            input_data = np.array(data['input'], dtype=np.float32)

        # 提交到线程池处理
        future = executor.submit(predict_async, input_data, client_ip)
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


@app.route('/predict_csv', methods=['POST'])  # 两个接口
def predict_csv():
    """处理CSV文件上传的预测请求"""
    # 获取客户端IP用于日志
    client_ip = request.remote_addr
    extra = {'client_ip': client_ip}
    logger.info(f"Received CSV prediction request", extra=extra)

    # 检查文件上传
    if 'file' not in request.files:
        logger.warning("No file part in request", extra=extra)
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file", extra=extra)
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 读取CSV文件
        csv_data = file.read().decode('utf-8')
        csv_reader = csv.reader(io.StringIO(csv_data))

        # 解析CSV数据
        rows = list(csv_reader)
        if not rows:
            raise ValueError("CSV file is empty")

        # 检查是否有标题行
        has_header = request.form.get('has_header', 'false').lower() == 'true'
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows

        # 转换为numpy数组
        input_data = np.array(data_rows, dtype=np.float32)

        # 提交到线程池处理
        future = executor.submit(predict_async, input_data, client_ip)
        predictions = future.result(timeout=30)  # 更长的超时时间

        # 转换为Python原生类型
        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()

        # 创建包含预测结果的CSV
        output_filename = f"predictions_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # 添加预测结果列
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # 写入标题行
            if has_header:
                writer.writerow(headers + ['prediction'])
            else:
                writer.writerow(headers + ['prediction'])

            # 写入数据和预测结果
            for i, row in enumerate(data_rows):
                pred = predictions[i] if predictions.ndim > 1 else predictions[i].tolist()
                if isinstance(pred, list):
                    # 对于多输出预测，创建多个列
                    pred_row = [f"pred_{j}" for j in range(len(pred))]
                    writer.writerow(row + pred_row)
                else:
                    writer.writerow(row + [str(pred)])

        logger.info(f"CSV prediction successful. Output file: {output_path}", extra=extra)

        # 返回生成的CSV文件
        return send_file(
            output_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=output_filename
        )

    except ValueError as ve:
        logger.error(f"CSV processing failed: {str(ve)}", extra=extra)
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.exception("Unexpected error during CSV prediction", extra=extra)
        return jsonify({'error': 'Internal server error'}), 500


def predict_async(input_data, client_ip):
    """异步处理预测任务"""
    extra = {'client_ip': client_ip}

    try:
        logger.debug("Starting input preprocessing", extra=extra)
        input_tensor = preprocess_data(input_data, input_shape=OM.input_shape)

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

