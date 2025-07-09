import requests
import numpy as np
import json
import time
import argparse


class ModelClient:  # 模型客户端
    def __init__(self, base_url="http://127.0.0.1:5000", timeout=10, max_retries=3, retry_delay=1):
        """
        初始化模型客户端

        参数:
            base_url (str): 服务器基础URL (例如: http://localhost:5000)
            timeout (int): 请求超时时间(秒)
            max_retries (int): 最大重试次数
            retry_delay (int): 重试延迟(秒)
        """
        self.base_url = base_url.rstrip('/')
        self.predict_url = f"{self.base_url}/predict"
        self.health_url = f"{self.base_url}/health"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    def check_health(self):  #
        """检查服务器健康状态"""
        try:
            response = self.session.get(self.health_url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return {
                'status': 'unhealthy',
                'error': f"HTTP {response.status_code}: {response.text}"
            }
        except Exception as e:
            return {
                'status': 'unreachable',
                'error': str(e)
            }

    def send_request(self, input_data, verbose=False):
        """
        发送预测请求

        参数:
            input_data: NumPy数组或可转换为NumPy数组的数据
            verbose (bool): 是否打印详细信息

        返回:
            tuple: (预测结果, 响应时间ms, 状态码) 或 (错误信息, 0, 状态码)
        """
        # 转换为NumPy数组（如果还不是）
        if not isinstance(input_data, np.ndarray):
            try:
                input_data = np.array(input_data)
            except Exception as e:
                return f"Invalid input data: {str(e)}", 0, 400

        # 准备请求数据
        payload = {
            'input': input_data.tolist()  # 转换为JSON可序列化格式
        }

        headers = {
            'Content-Type': 'application/json'
        }

        # 重试机制
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                # 发送请求
                response = self.session.post(
                    self.predict_url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=self.timeout
                )

                elapsed = (time.perf_counter() - start_time) * 1000  # ms

                if verbose:
                    print(f"Request sent | Shape: {input_data.shape} | "
                          f"Size: {input_data.size} elements | "
                          f"Type: {input_data.dtype}")

                # 处理响应
                if response.status_code == 200:
                    result = response.json()
                    if verbose:
                        print(f"Response received | Status: {response.status_code} | "
                              f"Time: {elapsed:.2f}ms")
                    return result['prediction'], elapsed, response.status_code

                # 处理错误响应
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', error_msg)
                except:
                    pass

                if verbose:
                    print(f"Error response | Status: {response.status_code} | "
                          f"Message: {error_msg}")

                # 如果是服务器错误且未达到最大重试次数，则重试
                if 500 <= response.status_code < 600 and attempt < self.max_retries:
                    if verbose:
                        print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue

                return error_msg, elapsed, response.status_code

            except requests.exceptions.Timeout:  # 超时处理
                elapsed = (time.perf_counter() - start_time) * 1000
                if verbose:
                    print(f"Request timed out after {elapsed:.2f}ms")
                if attempt < self.max_retries:
                    if verbose:
                        print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                return "Request timeout", elapsed, 504

            except requests.exceptions.RequestException as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                return f"Request failed: {str(e)}", elapsed, 500

        return "Max retries exceeded", 0, 503


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='TensorFlow Model Client')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='Server base URL')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Request timeout in seconds')
    parser.add_argument('--shape', type=str, default='672, 12',
                        help='Input shape (e.g., "10" for 1D, "3,224,224" for 3D)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'int32', 'float64'],
                        help='Input data type')
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of samples to send')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    # 创建客户端
    client = ModelClient(
        base_url=args.url,
        timeout=args.timeout
    )

    # 检查服务器健康状态
    health = client.check_health()
    print("\n===== Server Health Check =====")
    print(json.dumps(health, indent=2))

    if health.get('status') != 'healthy':
        print("\n⚠️ Server is not healthy. Aborting prediction.")
        return

    # 解析输入形状
    try:
        shape = tuple(map(int, args.shape.split(',')))
        if len(shape) == 1:
            shape = shape[0]  # 对于1D数组
    except:
        print(f"Invalid shape format: {args.shape}")
        return

    # 创建测试数据
    input_data = np.random.randn(*((args.samples,) + shape
                                   if isinstance(shape, tuple)
                                   else (args.samples, shape)))

    # 转换数据类型
    dtype_map = {
        'float32': np.float32,
        'int32': np.int32,
        'float64': np.float64
    }
    input_data = input_data.astype(dtype_map[args.dtype])

    print(f"\n===== Sending Prediction Request =====")
    print(f"Input shape: {input_data.shape}")
    print(f"Data type: {input_data.dtype}")
    print(f"Samples: {args.samples}")

    # 发送请求
    result, elapsed, status = client.send_request(input_data, verbose=args.verbose)

    print(f"\n===== Response Summary =====")
    print(f"Status Code: {status}")
    print(f"Response Time: {elapsed:.2f}ms")

    if status == 200:
        # 成功响应处理
        if isinstance(result, list) and len(result) > 0:
            sample_result = result[0] if isinstance(result[0], list) else result[:5]

            print(f"\nPrediction Result Sample:")
            if isinstance(sample_result, list) and len(sample_result) > 5:
                print(f"  First 5 elements: {sample_result[:5]}")
            else:
                print(f"  {sample_result}")

            print(f"\nResult Type: {type(result)}")
            if isinstance(result, list):
                print(f"Result Length: {len(result)}")
                if len(result) > 0 and isinstance(result[0], list):
                    print(f"First Sample Length: {len(result[0])}")
        else:
            print(f"\nPrediction Result: {result}")
    else:
        # 错误处理
        print(f"\nError: {result}")


if __name__ == '__main__':
    main()