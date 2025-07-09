import requests
import numpy as np
import json
import time
import argparse
import os
import csv
import pandas as pd
import uuid


class ModelClient:
    def __init__(self, base_url, timeout=10, max_retries=3, retry_delay=1):
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
        self.predict_csv_url = f"{self.base_url}/predict_csv"
        self.health_url = f"{self.base_url}/health"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def check_health(self):
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

    def predict_from_csv(self, file_path, has_header=True, output_dir=None, verbose=False):
        """
        通过CSV文件进行预测

        参数:
            file_path (str): CSV文件路径
            has_header (bool): CSV是否有标题行
            output_dir (str): 结果保存目录
            verbose (bool): 是否打印详细信息

        返回:
            str: 预测结果CSV文件路径
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # 准备请求数据
        files = {'file': open(file_path, 'rb')}
        data = {'has_header': 'true' if has_header else 'false'}

        # 重试机制
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                # 发送请求
                response = self.session.post(
                    self.predict_csv_url,
                    files=files,
                    data=data,
                    timeout=self.timeout * 3  # 更长的超时时间
                )

                elapsed = (time.perf_counter() - start_time) * 1000  # ms

                if verbose:
                    print(f"CSV file sent | Size: {os.path.getsize(file_path) / 1024:.2f}KB")

                # 处理响应
                if response.status_code == 200:
                    # 保存结果文件
                    if not output_dir:
                        output_dir = os.path.dirname(file_path)

                    output_filename = f"predictions_{uuid.uuid4().hex[:8]}.csv"
                    output_path = os.path.join(output_dir, output_filename)

                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    if verbose:
                        print(f"Response received | Status: {response.status_code} | "
                              f"Time: {elapsed:.2f}ms")
                        print(f"Results saved to: {output_path}")

                    return output_path

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

                raise RuntimeError(f"Server error: {error_msg}")

            except requests.exceptions.Timeout:
                elapsed = (time.perf_counter() - start_time) * 1000
                if verbose:
                    print(f"Request timed out after {elapsed:.2f}ms")
                if attempt < self.max_retries:
                    if verbose:
                        print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                raise TimeoutError("Request timeout")

            except requests.exceptions.RequestException as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                raise ConnectionError(f"Request failed: {str(e)}")

            finally:
                files['file'].close()

        raise RuntimeError("Max retries exceeded")

    def predict_from_sources(self, sources, verbose=False):
        """
        从多个数据源进行预测

        参数:
            sources (list): 数据源配置列表
            verbose (bool): 是否打印详细信息

        返回:
            dict: 预测结果
        """
        # 准备请求数据
        payload = {'sources': sources}

        # 重试机制
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                # 发送请求
                response = self.session.post(
                    self.predict_url,
                    json=payload,
                    timeout=self.timeout
                )

                elapsed = (time.perf_counter() - start_time) * 1000  # ms

                if verbose:
                    print(f"Multi-source request sent | Sources: {len(sources)}")

                # 处理响应
                if response.status_code == 200:
                    result = response.json()
                    if verbose:
                        print(f"Response received | Status: {response.status_code} | "
                              f"Time: {elapsed:.2f}ms")
                    return result['prediction']

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

                raise RuntimeError(f"Server error: {error_msg}")

            except requests.exceptions.Timeout:
                elapsed = (time.perf_counter() - start_time) * 1000
                if verbose:
                    print(f"Request timed out after {elapsed:.2f}ms")
                if attempt < self.max_retries:
                    if verbose:
                        print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                raise TimeoutError("Request timeout")

            except requests.exceptions.RequestException as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                raise ConnectionError(f"Request failed: {str(e)}")

        raise RuntimeError("Max retries exceeded")

    def send_request(self, input_data, verbose=False):
        """
        发送标准预测请求

        参数:
            input_data: NumPy数组或可转换为NumPy数组的数据
            verbose (bool): 是否打印详细信息

        返回:
            list: 预测结果
        """
        # 转换为NumPy数组（如果还不是）
        if not isinstance(input_data, np.ndarray):
            try:
                input_data = np.array(input_data)
            except Exception as e:
                raise ValueError(f"Invalid input data: {str(e)}")

        # 准备请求数据
        payload = {
            'input': input_data.tolist()  # 转换为JSON可序列化格式
        }

        # 重试机制
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                # 发送请求
                response = self.session.post(
                    self.predict_url,
                    json=payload,
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
                    return result['prediction']

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

                raise RuntimeError(f"Server error: {error_msg}")

            except requests.exceptions.Timeout:
                elapsed = (time.perf_counter() - start_time) * 1000
                if verbose:
                    print(f"Request timed out after {elapsed:.2f}ms")
                if attempt < self.max_retries:
                    if verbose:
                        print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                raise TimeoutError("Request timeout")

            except requests.exceptions.RequestException as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                raise ConnectionError(f"Request failed: {str(e)}")

        raise RuntimeError("Max retries exceeded")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Enhanced Model Client')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='Server base URL')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'csv', 'multi-source'],
                        help='Prediction mode')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Request timeout in seconds')
    parser.add_argument('--shape', type=str, default='672, 12',
                        help='Input shape (e.g., "10" for 1D, "3,224,224" for 3D)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'int32', 'float64'],
                        help='Input data type')
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of samples to send')
    parser.add_argument('--csv', type=str,
                        help='Path to CSV file for prediction')
    parser.add_argument('--has_header', action='store_true',
                        help='CSV file has header row')
    parser.add_argument('--api_source', type=str,
                        help='External API URL for features')
    parser.add_argument('--api_params', type=str, default='{}',
                        help='API parameters as JSON string')
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

    # 根据模式执行预测
    if args.mode == 'csv':
        # CSV模式
        if not args.csv:
            print("Error: --csv argument is required for CSV mode")
            return

        print(f"\n===== CSV Prediction Mode =====")
        print(f"CSV File: {args.csv}")
        print(f"Has Header: {args.has_header}")

        try:
            result_path = client.predict_from_csv(
                args.csv,
                has_header=args.has_header,
                verbose=args.verbose
            )
            print(f"\n✅ Prediction results saved to: {result_path}")

            # 显示结果预览
            if args.verbose:
                df = pd.read_csv(result_path)
                print("\nResults Preview:")
                print(df.head())

        except Exception as e:
            print(f"\n❌ Prediction failed: {str(e)}")

    elif args.mode == 'multi-source':
        # 多源模式
        print("\n===== Multi-Source Prediction Mode =====")

        sources = []

        # 添加直接特征源
        try:
            # 解析输入形状
            shape = tuple(map(int, args.shape.split(',')))
            if len(shape) == 1:
                shape = shape[0]  # 对于1D数组

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

            sources.append({
                'type': 'direct',
                'features': input_data.tolist()
            })
            print(f"Added direct source | Shape: {input_data.shape}")
        except Exception as e:
            print(f"Error creating direct source: {str(e)}")
            return

        # 添加API源（如果提供）
        if args.api_source:
            try:
                api_params = json.loads(args.api_params)
                sources.append({
                    'type': 'external',
                    'config': {
                        'type': 'api',
                        'url': args.api_source,
                        'params': api_params
                    }
                })
                print(f"Added API source | URL: {args.api_source}")
            except Exception as e:
                print(f"Error adding API source: {str(e)}")

        try:
            result = client.predict_from_sources(sources, verbose=args.verbose)
            print("\n===== Prediction Result =====")

            # 显示结果样本
            if isinstance(result, list) and len(result) > 0:
                sample_result = result[0] if isinstance(result[0], list) else result[:5]

                print(f"Result Sample:")
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
                print(f"Result: {result}")

        except Exception as e:
            print(f"\n❌ Prediction failed: {str(e)}")

    else:
        # 标准模式
        print("\n===== Standard Prediction Mode =====")

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

        print(f"Input shape: {input_data.shape}")
        print(f"Data type: {input_data.dtype}")
        print(f"Samples: {args.samples}")

        try:
            result = client.send_request(input_data, verbose=args.verbose)
            print("\n===== Prediction Result =====")

            # 显示结果样本
            if isinstance(result, list) and len(result) > 0:
                sample_result = result[0] if isinstance(result[0], list) else result[:5]

                print(f"Result Sample:")
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
                print(f"Result: {result}")

        except Exception as e:
            print(f"\n❌ Prediction failed: {str(e)}")


if __name__ == '__main__':
    main()