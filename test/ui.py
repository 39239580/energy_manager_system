import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import time
import holidays
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="NeuroGrid Pro - 智能能源管理平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 自定义CSS样式
def set_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #0f4c81;
            --secondary-color: #00c7c7;
            --accent-color: #ff6b6b;
            --strategy-color: #9b5de5;
            --revenue-color: #2ec4b6;
            --background-color: #0e1117;
            --card-color: #1a1a2e;
            --text-color: #ffffff;
            --border-color: #2d4059;
        }

        body {
            background: radial-gradient(circle at top right, var(--background-color) 0%, #16213e 100%);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .main {
            background-color: rgba(26, 26, 46, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 199, 199, 0.2);
            border: 1px solid rgba(0, 199, 199, 0.18);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--secondary-color);
            background: linear-gradient(90deg, var(--secondary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 0.3rem;
            font-weight: 700;
        }

        .stButton>button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 30px;
            padding: 0.7rem 1.5rem;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 199, 199, 0.3);
        }

        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0, 199, 199, 0.4);
        }

        .stTextInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div {
            background-color: rgba(30, 33, 48, 0.7);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }

        .stDateInput>div>div>input {
            background-color: rgba(30, 33, 48, 0.7);
            color: var(--text-color);
        }

        .stProgress>div>div>div>div {
            background: linear-gradient(90deg, var(--secondary-color), var(--accent-color));
        }

        .card {
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 199, 199, 0.4);
        }

        .metric-card {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(0, 199, 199, 0.3);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }

        .metric-card:hover .metric-value {
            color: var(--secondary-color);
        }

        .metric-label {
            font-size: 0.9rem;
            color: #a0aec0;
        }

        .glowing-border {
            border: 1px solid var(--secondary-color);
            box-shadow: 0 0 10px rgba(0, 199, 199, 0.5);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .glowing-border:hover {
            box-shadow: 0 0 20px rgba(0, 199, 199, 0.8);
        }

        .strategy-border {
            border: 1px solid var(--strategy-color);
            box-shadow: 0 0 10px rgba(155, 93, 229, 0.5);
        }

        .strategy-border:hover {
            box-shadow: 0 0 20px rgba(155, 93, 229, 0.8);
        }

        .revenue-border {
            border: 1px solid var(--revenue-color);
            box-shadow: 0 0 10px rgba(46, 196, 182, 0.5);
        }

        .revenue-border:hover {
            box-shadow: 0 0 20px rgba(46, 196, 182, 0.8);
        }

        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--secondary-color), transparent);
            margin: 1.5rem 0;
        }

        .footer {
            text-align: center;
            padding: 1rem;
            color: #a0aec0;
            font-size: 0.9rem;
            margin-top: 2rem;
            border-top: 1px solid rgba(45, 64, 89, 0.5);
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 199, 199, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 199, 199, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 199, 199, 0); }
        }

        .strategy-pulse {
            animation: strategyPulse 2s infinite;
        }

        @keyframes strategyPulse {
            0% { box-shadow: 0 0 0 0 rgba(155, 93, 229, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(155, 93, 229, 0); }
            100% { box-shadow: 0 0 0 0 rgba(155, 93, 229, 0); }
        }

        .revenue-pulse {
            animation: revenuePulse 2s infinite;
        }

        @keyframes revenuePulse {
            0% { box-shadow: 0 0 0 0 rgba(46, 196, 182, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(46, 196, 182, 0); }
            100% { box-shadow: 0 0 0 0 rgba(46, 196, 182, 0); }
        }

        .tab-content {
            padding: 1.5rem 0;
        }

        .grid-flow {
            position: relative;
            overflow: hidden;
        }

        .grid-flow::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 199, 199, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 199, 199, 0.1) 1px, transparent 1px);
            background-size: 30px 30px;
            z-index: -1;
        }

        .energy-flow {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
        }

        .energy-node {
            position: absolute;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--secondary-color);
            box-shadow: 0 0 10px var(--secondary-color);
            animation: energyFlow 5s infinite linear;
        }

        @keyframes energyFlow {
            0% {
                transform: translateY(-10px);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(calc(100% + 10px));
                opacity: 0;
            }
        }

        .fade-in {
            animation: fadeIn 1s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .slide-in-left {
            animation: slideInLeft 0.8s ease-out;
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .slide-in-right {
            animation: slideInRight 0.8s ease-out;
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .pulse-text {
            animation: pulseText 2s infinite;
        }

        @keyframes pulseText {
            0% { text-shadow: 0 0 5px rgba(0, 199, 199, 0.7); }
            50% { text-shadow: 0 0 15px rgba(0, 199, 199, 1); }
            100% { text-shadow: 0 0 5px rgba(0, 199, 199, 0.7); }
        }

        .energy-wave {
            position: relative;
            overflow: hidden;
            border-radius: 12px;
        }

        .energy-wave::after {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 199, 199, 0.2) 0%, transparent 70%);
            animation: wave 8s infinite linear;
            z-index: -1;
        }

        @keyframes wave {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }

        .dashboard-header {
            background: linear-gradient(90deg, rgba(15, 76, 129, 0.7), rgba(0, 199, 199, 0.7));
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .dashboard-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%2300c7c7' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
            opacity: 0.1;
        }

        /* 经济收益页面优化 */
        .financial-section h4 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #2ec4b6;
        }

        .financial-metric .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .financial-metric .metric-label {
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }

        .revenue-detail {
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 0.8rem;
        }

        .revenue-detail strong {
            font-weight: 700;
            color: #2ec4b6;
        }

        .revenue-highlight {
            font-size: 1.3rem;
            font-weight: 700;
            color: #f9c74f;
            margin: 1rem 0;
        }

        .revenue-card {
            padding: 1.8rem;
            background: linear-gradient(135deg, rgba(26, 26, 46, 0.9), rgba(22, 33, 62, 0.9));
        }

        .revenue-card h4 {
            border-bottom: 2px solid #2ec4b6;
            padding-bottom: 0.8rem;
            margin-bottom: 1.5rem;
        }

        .revenue-breakdown {
            background: rgba(30, 33, 48, 0.7);
            border-radius: 10px;
            padding: 1.2rem;
            margin-top: 1.5rem;
        }

        .revenue-breakdown h5 {
            color: #00c7c7;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }

        .revenue-breakdown p {
            font-size: 1.1rem;
            margin-bottom: 0.6rem;
            padding-left: 1rem;
            position: relative;
        }

        .revenue-breakdown p:before {
            content: "•";
            position: absolute;
            left: 0;
            color: #2ec4b6;
            font-size: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


set_custom_css()


# 创建能源流动动画
def create_energy_flow():
    st.markdown("""
    <div class="energy-flow">
        <style>
            .energy-node {
                animation-duration: 5s;
                animation-delay: calc(var(--delay) * 0.1s);
            }
        </style>
    """, unsafe_allow_html=True)

    # 创建多个节点
    for i in range(30):
        left = np.random.randint(0, 100)
        delay = np.random.randint(0, 20)
        st.markdown(
            f'<div class="energy-node" style="--delay: {delay}; left: {left}%;"></div>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# 加载模型和Scaler
@st.cache_resource
def load_resources():
    try:
        # 在实际应用中替换为真实模型路径
        model = tf.keras.models.load_model('best_model_cnn_bilstm.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        # 如果模型不存在，返回None
        return None, None


model, scaler = load_resources()


# 电价预测模型（模拟）
class ElectricityPriceModel:
    def __init__(self):
        self.peak_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.off_peak_hours = [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23]
        self.base_price = 0.6  # 基础电价
        self.peak_multiplier = 1.5  # 高峰时段乘数
        self.off_peak_multiplier = 0.7  # 低谷时段乘数

    def predict_price(self, timestamp):
        """预测电价"""
        hour = timestamp.hour
        if hour in self.peak_hours:
            return self.base_price * self.peak_multiplier
        elif hour in self.off_peak_hours:
            return self.base_price * self.off_peak_multiplier
        else:
            return self.base_price * 1.2  # 平段电价

    def predict_future_prices(self, timestamps):
        """预测未来电价序列"""
        return [self.predict_price(ts) for ts in timestamps]


# 工商业运行策略
class BusinessStrategyController:
    def __init__(self):
        self.strategies = {
            "峰谷套利": {
                "description": "在低谷时段充电，高峰时段放电，利用电价差获利",
                "parameters": ["储能容量", "充放电效率", "最大充放电功率"],
                "color": "#9b5de5",
                "icon": "🔋"
            },
            "需量控制": {
                "description": "控制最大需量，避免需量电费过高",
                "parameters": ["需量阈值", "响应速度", "控制精度"],
                "color": "#00c7c7",
                "icon": "📉"
            },
            "需求响应": {
                "description": "响应电网调度指令，获取补贴收益",
                "parameters": ["响应容量", "响应速度", "最小持续时间"],
                "color": "#ff6b6b",
                "icon": "📡"
            },
            "新能源消纳": {
                "description": "配合光伏发电，提高自发自用率",
                "parameters": ["光伏容量", "预测精度", "消纳目标"],
                "color": "#2ec4b6",
                "icon": "☀️"
            }
        }

        self.default_values = {
            "储能容量": 1000,  # kWh
            "充放电效率": 0.92,
            "最大充放电功率": 500,  # kW
            "需量阈值": 800,  # kW
            "响应速度": 0.5,  # 秒
            "控制精度": 0.95,
            "响应容量": 300,  # kW
            "最小持续时间": 2,  # 小时
            "光伏容量": 200,  # kW
            "预测精度": 0.85,
            "消纳目标": 0.9
        }

    def get_strategy_params(self, strategy_name):
        """获取策略参数"""
        strategy = self.strategies[strategy_name]
        params = {}
        for param in strategy["parameters"]:
            params[param] = self.default_values[param]
        return params

    def calculate_revenue(self, strategy_name, params, load_data, price_data):
        """计算策略收益（模拟）"""
        # 简化版收益计算
        if strategy_name == "峰谷套利":
            # 找出低谷和高峰时段
            min_idx = np.argmin(price_data)
            max_idx = np.argmax(price_data)
            min_price = price_data[min_idx]
            max_price = price_data[max_idx]

            # 确保低谷在高峰之前
            if min_idx > max_idx:
                min_idx = np.argmin(price_data[:max_idx])
                min_price = price_data[min_idx]

            # 计算套利收益
            energy = min(params["储能容量"], params["最大充放电功率"] * 1)  # 假设1小时充放电
            revenue = energy * (max_price - min_price) * params["充放电效率"]
            return revenue

        elif strategy_name == "需量控制":
            # 计算避免的需量费用
            max_load = max(load_data)
            threshold = params["需量阈值"]
            if max_load > threshold:
                avoided_demand = max_load - threshold
                # 假设需量电费为50元/kW/月
                revenue = avoided_demand * 50
            else:
                revenue = 0
            return revenue

        elif strategy_name == "需求响应":
            # 计算响应补贴
            # 假设每次响应补贴为3元/kW
            revenue = params["响应容量"] * 3
            return revenue

        elif strategy_name == "新能源消纳":
            # 计算节省的电费
            # 假设光伏发电量为200kWh，节省电费为发电量*平均电价
            pv_generation = 200  # kWh
            avg_price = sum(price_data) / len(price_data)
            revenue = pv_generation * avg_price * params["消纳目标"]
            return revenue

        return 0


# 经济收益分析
class EconomicAnalysis:
    def __init__(self):
        self.cost_factors = {
            "储能投资成本": 1500,  # 元/kWh
            "运维成本": 0.05,  # 元/kWh
            "电价": 0.6,  # 元/kWh
            "需量电费": 50,  # 元/kW/月
            "需求响应补贴": 3  # 元/kW/次
        }

    def calculate_roi(self, revenue, cost):
        """计算投资回报率"""
        if cost == 0:
            return float('inf')
        return revenue / cost * 100

    def calculate_payback_period(self, investment, annual_savings):
        """计算投资回收期"""
        if annual_savings <= 0:
            return float('inf')
        return investment / annual_savings

    def generate_financial_report(self, strategy_name, revenue, params):
        """生成财务报告"""
        # 计算投资成本
        if strategy_name == "峰谷套利":
            investment = params["储能容量"] * self.cost_factors["储能投资成本"]
            annual_savings = revenue * 365  # 假设每天执行一次
        elif strategy_name == "需量控制":
            investment = 0  # 假设使用现有设备
            annual_savings = revenue * 12  # 每月节省
        elif strategy_name == "需求响应":
            investment = 0  # 假设使用现有设备
            annual_savings = revenue * 50  # 假设每年响应50次
        elif strategy_name == "新能源消纳":
            investment = params["光伏容量"] * 4000  # 光伏系统投资（元/kW）
            annual_savings = revenue * 365  # 每天收益

        # 计算财务指标
        roi = self.calculate_roi(annual_savings, investment)
        payback_years = self.calculate_payback_period(investment, annual_savings)

        return {
            "strategy": strategy_name,
            "investment": investment,
            "annual_savings": annual_savings,
            "roi": roi,
            "payback_years": payback_years
        }


# 生成模拟数据（缓存）
@st.cache_data
def generate_synthetic_load(time_points):
    base_load = 100 + 50 * np.sin(2 * np.pi * time_points / (24 * 96))
    weekly_pattern = 20 * np.sin(2 * np.pi * time_points / (7 * 24 * 96))
    seasonal_pattern = 30 * np.sin(2 * np.pi * time_points / (365 * 96))
    workday_pattern = np.zeros_like(time_points)
    for i in range(len(time_points)):
        day_of_week = (i // 96) % 7
        if day_of_week < 5:  # 工作日
            workday_pattern[i] = 15 * np.sin(2 * np.pi * (i % 96) / 96)
        else:  # 周末
            workday_pattern[i] = 10 * np.sin(2 * np.pi * (i % 96) / 96)

    # 添加随机事件
    random_events = np.zeros_like(time_points)
    event_indices = np.random.choice(len(time_points), size=50, replace=False)
    random_events[event_indices] = np.random.uniform(20, 50, size=50)

    noise = 8 * np.random.normal(size=len(time_points))
    load_data = base_load + weekly_pattern + seasonal_pattern + workday_pattern + random_events + noise
    return np.maximum(load_data, 10)


@st.cache_data
def add_cyclic_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
    df['is_weekend'] = df['timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    return df


@st.cache_data
def add_holiday_features(df):
    cn_holidays = holidays.CountryHoliday('CN')
    df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in cn_holidays else 0)
    return df


@st.cache_data
def add_weather_data(df):
    np.random.seed(42)
    days = len(df) // 96

    # 温度模拟
    base_temp = 15 + 15 * np.sin(2 * np.pi * np.arange(days) / 365)
    daily_temp_variation = 8 * np.sin(2 * np.pi * np.arange(96) / 96)

    # 添加随机天气波动
    temp_anomaly = np.zeros(days)
    for i in range(0, days, 30):
        if np.random.rand() > 0.7:
            event_duration = np.random.randint(3, 10)
            event_intensity = np.random.uniform(5, 15)
            temp_anomaly[i:i + event_duration] = event_intensity

    temp_data = np.zeros(len(df))
    for i in range(days):
        start_idx = i * 96
        end_idx = (i + 1) * 96
        temp_data[start_idx:end_idx] = base_temp[i] + daily_temp_variation + temp_anomaly[i] + 2 * np.random.normal(
            size=96)

    # 湿度模拟
    humidity = 70 - 0.5 * (temp_data - 20) + 5 * np.random.normal(size=len(df))
    humidity = np.clip(humidity, 30, 95)

    df['temperature'] = temp_data
    df['humidity'] = humidity
    return df


# 生成模拟数据（缓存）
@st.cache_data
def create_synthetic_data():
    time_points = np.arange(0, 365 * 96)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=15 * i) for i in range(len(time_points))]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'load': generate_synthetic_load(time_points)
    })
    df = add_cyclic_features(df)
    df = add_holiday_features(df)
    df = add_weather_data(df)

    # 添加电价预测
    price_model = ElectricityPriceModel()
    df['electricity_price'] = df['timestamp'].apply(price_model.predict_price)

    return df


# 创建科幻风格图表
def create_scifi_chart(df, title, y_title="负荷 (kW)"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 添加负荷曲线
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['load'],
        mode='lines',
        name='负荷',
        line=dict(color='#00c7c7', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 199, 199, 0.1)'
    ))

    # 添加温度曲线
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines',
        name='温度',
        line=dict(color='#ff6b6b', width=2, dash='dot'),
        yaxis='y2'
    ))

    # 添加电价曲线
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['electricity_price'],
        mode='lines',
        name='电价',
        line=dict(color='#f9c74f', width=2, dash='dash'),
        yaxis='y3'
    ))

    # 布局配置
    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color='#00c7c7')),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='时间',
            gridcolor='rgba(100, 100, 100, 0.2)',
            linecolor='#2d4059',
            showgrid=True
        ),
        yaxis=dict(
            title=y_title,
            gridcolor='rgba(100, 100, 100, 0.2)',
            linecolor='#2d4059',
            showgrid=True
        ),
        yaxis2=dict(
            title='温度 (°C)',
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)',
            showgrid=False
        ),
        yaxis3=dict(
            title='电价 (元/kWh)',
            overlaying='y',
            side='right',
            anchor='free',
            position=1.0,
            gridcolor='rgba(0,0,0,0)',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )

    # 添加网格线
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(45, 64, 89, 0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(45, 64, 89, 0.5)')

    return fig


# 创建策略执行图
def create_strategy_chart(load_data, price_data, strategy_name, action_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 添加负荷曲线
    fig.add_trace(go.Scatter(
        x=np.arange(len(load_data)),
        y=load_data,
        mode='lines',
        name='负荷',
        line=dict(color='#00c7c7', width=3)
    ))

    # 添加电价曲线
    fig.add_trace(go.Scatter(
        x=np.arange(len(price_data)),
        y=price_data,
        mode='lines',
        name='电价',
        line=dict(color='#f9c74f', width=2, dash='dash'),
        yaxis='y2'
    ))

    # 添加策略动作
    action_colors = {
        "充电": "#2ec4b6",
        "放电": "#ff6b6b",
        "待机": "#9b5de5"
    }

    for action in set(action_data):
        indices = [i for i, a in enumerate(action_data) if a == action]
        fig.add_trace(go.Scatter(
            x=indices,
            y=[load_data[i] for i in indices],
            mode='markers',
            name=action,
            marker=dict(color=action_colors.get(action, "#ffffff"), size=8)
        ))

    # 布局配置
    fig.update_layout(
        title=dict(text=f"{strategy_name}策略执行图", font=dict(size=22, color='#9b5de5')),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='时间点 (15分钟间隔)',
            gridcolor='rgba(100, 100, 100, 0.2)',
            linecolor='#2d4059',
            showgrid=True
        ),
        yaxis=dict(
            title='负荷 (kW)',
            gridcolor='rgba(100, 100, 100, 0.2)',
            linecolor='#2d4059',
            showgrid=True
        ),
        yaxis2=dict(
            title='电价 (元/kWh)',
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )

    return fig


# 创建财务分析图
def create_financial_chart(report):
    fig = go.Figure()

    # 投资和收益柱状图
    fig.add_trace(go.Bar(
        x=['投资', '年收益'],
        y=[report['investment'], report['annual_savings']],
        name='金额',
        marker_color=['#9b5de5', '#2ec4b6']
    ))

    # 添加ROI和回收期
    fig.add_trace(go.Scatter(
        x=['投资回报率', '回收期'],
        y=[report['roi'], report['payback_years']],
        mode='markers+text',
        name='财务指标',
        marker=dict(size=15, color='#f9c74f'),
        text=[f"{report['roi']:.1f}%", f"{report['payback_years']:.1f}年"],
        textposition='top center',
        textfont=dict(color='white', size=14),
        yaxis='y2'
    ))

    # 布局配置
    fig.update_layout(
        title=dict(text=f"{report['strategy']}财务分析", font=dict(size=22, color='#2ec4b6')),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='指标',
            gridcolor='rgba(100, 100, 100, 0.2)',
            linecolor='#2d4059',
            showgrid=True
        ),
        yaxis=dict(
            title='金额 (元)',
            gridcolor='rgba(100, 100, 100, 0.2)',
            linecolor='#2d4059',
            showgrid=True
        ),
        yaxis2=dict(
            title='百分比/年',
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# 创建3D能源流动图
def create_3d_energy_flow():
    # 创建网格
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # 创建3D曲面图
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])

    # 添加能量流动点
    for _ in range(100):
        x_point = np.random.uniform(-5, 5)
        y_point = np.random.uniform(-5, 5)
        z_point = np.sin(np.sqrt(x_point ** 2 + y_point ** 2))
        fig.add_trace(go.Scatter3d(
            x=[x_point],
            y=[y_point],
            z=[z_point],
            mode='markers',
            marker=dict(
                size=5,
                color='#00c7c7',
                opacity=0.8
            )
        ))

    fig.update_layout(
        title='3D能源流动网络',
        template='plotly_dark',
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=400
    )

    return fig


# 主界面
def main():
    # 添加能源流动背景
    create_energy_flow()

    # 顶部标题区域
    with st.container():
        st.markdown("""
        <div class="dashboard-header fade-in">
            <h1 class="pulse-text">⚡ NeuroGrid Pro - 智能能源管理平台</h1>
            <p>基于深度学习的能源管理系统，集负荷预测、电价预测、运行策略优化和经济收益分析于一体</p>
        </div>
        """, unsafe_allow_html=True)

    # 创建选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 数据探索", "🔌 电价预测", "🔮 负荷预测", "⚙️ 策略控制", "💰 经济收益"])

    # 创建模拟数据（使用缓存）
    with st.spinner('正在准备数据...'):
        df = create_synthetic_data()

    price_model = ElectricityPriceModel()  #电价预测模型
    strategy_controller = BusinessStrategyController()  # 工商业策略控制
    economic_analysis = EconomicAnalysis()  # 经济模型

    # 其他标签页代码保持不变...
    with tab1:
        st.session_state.current_tab = 0
        st.subheader("历史能源数据")

        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("""
        <div class="metric-card energy-wave slide-in-left">
            <div class="metric-label">平均负荷</div>
            <div class="metric-value">128.5 kW</div>
            <div class="metric-label">±5.2%</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.2s;">
            <div class="metric-label">峰值负荷</div>
            <div class="metric-value">248.3 kW</div>
            <div class="metric-label">昨天 14:30</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.4s;">
            <div class="metric-label">平均电价</div>
            <div class="metric-value">0.72 ¥/kWh</div>
            <div class="metric-label">峰谷差 0.48</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.6s;">
            <div class="metric-label">总能耗</div>
            <div class="metric-value">1.12 GWh</div>
            <div class="metric-label">同比 -3.5%</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = create_scifi_chart(df.tail(96 * 7), "过去一周的负荷、温度和电价数据")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            <div class="card slide-in-right">
                <h4>数据统计</h4>
                <div class="divider"></div>
                <p>数据时间范围: 2023-01-01 至 2023-12-31</p>
                <p>数据点数: 35,040</p>
                <p>时间分辨率: 15分钟</p>
                <p>平均负荷: 128.5 kW</p>
                <p>平均电价: 0.72 元/kWh</p>
                <p>最高电价: 0.90 元/kWh</p>
                <p>最低电价: 0.42 元/kWh</p>
            </div>
            """, unsafe_allow_html=True)

        # 展示特征数据
        st.subheader("特征分析")
        features = st.multiselect(
            "选择要分析的特征",
            ['load', 'temperature', 'humidity', 'electricity_price',
             'hour_sin', 'hour_cos', 'is_weekend', 'is_holiday'],
            ['load', 'electricity_price', 'temperature']
        )

        if features:
            fig = px.line(
                df.melt(id_vars=['timestamp'], value_vars=features),
                x='timestamp',
                y='value',
                color='variable',
                template='plotly_dark',
                title='特征随时间变化',
                labels={'value': '数值', 'timestamp': '时间'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend_title_text='特征',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3D能源网络可视化
        st.subheader("能源网络拓扑")
        fig = create_3d_energy_flow()
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.session_state.current_tab = 1
        st.subheader("电价预测")

        # 关键指标卡片
        col1, col2, col3 = st.columns(3)
        col1.markdown("""
        <div class="metric-card energy-wave slide-in-left">
            <div class="metric-label">当前电价</div>
            <div class="metric-value">0.82 ¥/kWh</div>
            <div class="metric-label">高峰时段</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.2s;">
            <div class="metric-label">预测精度</div>
            <div class="metric-value">92.3%</div>
            <div class="metric-label">±0.05</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.4s;">
            <div class="metric-label">最大峰谷差</div>
            <div class="metric-value">0.48 ¥/kWh</div>
            <div class="metric-label">套利空间</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.slider(
                "预测天数",
                min_value=1, max_value=7, value=3, key="price_forecast_days"
            )

            if st.button("执行电价预测", key="price_forecast_btn", use_container_width=True):
                with st.spinner('电价预测中...'):
                    progress_bar = st.progress(0)

                    # 模拟预测过程
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)

                    # 生成预测时间点
                    last_timestamp = df['timestamp'].iloc[-1]
                    prediction_timestamps = [last_timestamp + timedelta(minutes=15 * (i + 1)) for i in
                                             range(forecast_days * 96)]

                    # 预测电价
                    predicted_prices = price_model.predict_future_prices(prediction_timestamps)

                    # 创建结果DataFrame
                    price_df = pd.DataFrame({
                        'timestamp': prediction_timestamps,
                        'predicted_price': predicted_prices
                    })

                    # 合并历史数据
                    historical_df = df[['timestamp', 'electricity_price']].rename(
                        columns={'electricity_price': 'actual_price'})
                    merged_df = pd.merge(price_df, historical_df, on='timestamp', how='left')

                    # 显示结果
                    st.success("电价预测完成！")

                    # 绘制预测结果
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=merged_df['timestamp'],
                        y=merged_df['actual_price'],
                        mode='lines',
                        name='历史电价',
                        line=dict(color='#00c7c7', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=merged_df['timestamp'],
                        y=merged_df['predicted_price'],
                        mode='lines',
                        name='预测电价',
                        line=dict(color='#f9c74f', width=3, dash='dash')
                    ))

                    fig.update_layout(
                        title=dict(text=f"{forecast_days}天电价预测", font=dict(size=22, color='#f9c74f')),
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title='时间'),
                        yaxis=dict(title='电价 (元/kWh)'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            <div class="card strategy-border slide-in-right">
                <h4>电价预测说明</h4>
                <div class="divider"></div>
                <p>本系统基于深度学习模型预测未来电价走势，考虑以下因素：</p>
                <ul>
                    <li>历史电价模式</li>
                    <li>负荷预测结果</li>
                    <li>天气条件</li>
                    <li>市场供需关系</li>
                    <li>政策调控因素</li>
                </ul>
                <div class="divider"></div>
                <p>电价预测精度：92.3%</p>
                <p>预测频率：每15分钟更新</p>
            </div>

            <div class="card revenue-border slide-in-right" style="margin-top: 1.5rem;">
                <h4>电价结构分析</h4>
                <div class="divider"></div>
                <p><strong>峰时段 (8:00-20:00)</strong>: 0.90 元/kWh</p>
                <p><strong>谷时段 (0:00-8:00, 20:00-24:00)</strong>: 0.42 元/kWh</p>
                <p><strong>平时段</strong>: 0.72 元/kWh</p>
                <div class="divider"></div>
                <p>最大峰谷价差: 0.48 元/kWh</p>
                <p>平均电价: 0.68 元/kWh</p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.session_state.current_tab = 2
        st.subheader("负荷预测配置")

        # 关键指标卡片
        col1, col2, col3 = st.columns(3)
        col1.markdown("""
        <div class="metric-card energy-wave slide-in-left">
            <div class="metric-label">当前负荷</div>
            <div class="metric-value">158.3 kW</div>
            <div class="metric-label">+12.4%</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.2s;">
            <div class="metric-label">预测精度</div>
            <div class="metric-value">94.7%</div>
            <div class="metric-label">±3.5 kW</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.4s;">
            <div class="metric-label">预测偏差</div>
            <div class="metric-value">2.8%</div>
            <div class="metric-label">优于行业标准</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_mode = st.selectbox(
                "预测模式",
                ["多步预测", "递归预测", "单步预测"],
                index=0
            )

        with col2:
            look_back_days = st.slider(
                "历史数据天数",
                min_value=1, max_value=14, value=7
            )

        with col3:
            forecast_days = st.slider(
                "预测天数",
                min_value=1, max_value=7, value=1
            )

        # 预测按钮
        if st.button("执行负荷预测", key="load_forecast_btn", use_container_width=True):
            with st.spinner('神经网格计算中...'):
                progress_bar = st.progress(0)

                # 模拟预测过程
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)

                # 模拟预测结果
                last_load = df['load'].iloc[-1]
                prediction = np.array([last_load * (1 + 0.1 * np.sin(i / 10)) for i in range(forecast_days * 96)])

                # 实际值（模拟）
                actual_values = np.array([last_load * (1 + 0.08 * np.sin(i / 10)) for i in range(forecast_days * 96)])

                # 创建预测结果DataFrame
                last_timestamp = df['timestamp'].iloc[-1]
                prediction_timestamps = [last_timestamp + timedelta(minutes=15 * (i + 1)) for i in
                                         range(len(prediction))]

                result_df = pd.DataFrame({
                    'timestamp': prediction_timestamps,
                    '实际负荷': actual_values,
                    '预测负荷': prediction
                })

                # 计算评估指标
                rmse = np.sqrt(np.mean((result_df['实际负荷'] - result_df['预测负荷']) ** 2))
                mae = np.mean(np.abs(result_df['实际负荷'] - result_df['预测负荷']))
                mape = np.mean(np.abs((result_df['实际负荷'] - result_df['预测负荷']) / result_df['实际负荷'])) * 100

                # 显示结果
                st.success("负荷预测完成！")

                # 指标卡片
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-label">均方根误差</div>
                    <div class="metric-value">{rmse:.2f}</div>
                    <div class="metric-label">千瓦</div>
                </div>
                """, unsafe_allow_html=True)

                col2.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-label">平均绝对误差</div>
                    <div class="metric-value">{mae:.2f}</div>
                    <div class="metric-label">千瓦</div>
                </div>
                """, unsafe_allow_html=True)

                col3.markdown(f"""
                <div class="metric-card pulse">
                    <div class="metric-label">平均绝对百分比误差</div>
                    <div class="metric-value">{mape:.2f}</div>
                    <div class="metric-label">%</div>
                </div>
                """, unsafe_allow_html=True)

                # 预测图表
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result_df['timestamp'],
                    y=result_df['实际负荷'],
                    mode='lines',
                    name='实际负荷',
                    line=dict(color='#00c7c7', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=result_df['timestamp'],
                    y=result_df['预测负荷'],
                    mode='lines',
                    name='预测负荷',
                    line=dict(color='#ff6b6b', width=3, dash='dash')
                ))

                fig.update_layout(
                    title=dict(text=f"{forecast_days}天负荷预测结果", font=dict(size=22, color='#ff6b6b')),
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title='时间'),
                    yaxis=dict(title='负荷 (kW)'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.session_state.current_tab = 3
        st.subheader("工商业运行策略控制")

        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("""
        <div class="metric-card energy-wave slide-in-left">
            <div class="metric-label">策略收益</div>
            <div class="metric-value">¥2,380</div>
            <div class="metric-label">今日累计</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.2s;">
            <div class="metric-label">充放电次数</div>
            <div class="metric-value">24</div>
            <div class="metric-label">今日累计</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.4s;">
            <div class="metric-label">需量节省</div>
            <div class="metric-value">45 kW</div>
            <div class="metric-label">峰值削减</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown("""
        <div class="metric-card energy-wave slide-in-left" style="animation-delay: 0.6s;">
            <div class="metric-label">新能源消纳</div>
            <div class="metric-value">86%</div>
            <div class="metric-label">光伏利用率</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            # 策略选择
            selected_strategy = st.selectbox(
                "选择优化策略",
                list(strategy_controller.strategies.keys()),
                index=0
            )

            # 显示策略描述
            strategy_info = strategy_controller.strategies[selected_strategy]
            st.markdown(f"""
            <div class="card strategy-border slide-in-left">
                <h4>{strategy_info["icon"]} {selected_strategy}策略</h4>
                <div class="divider"></div>
                <p>{strategy_info['description']}</p>
                <div class="divider"></div>
                <p><strong>适用场景:</strong></p>
                <ul>
                    <li>高电价差地区</li>
                    <li>需量电费高的企业</li>
                    <li>参与需求响应项目</li>
                    <li>配备光伏发电系统</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # 策略参数配置
            st.subheader("策略参数配置")
            params = strategy_controller.get_strategy_params(selected_strategy)
            for param in strategy_info['parameters']:
                params[param] = st.slider(
                    param,
                    min_value=0.0,
                    max_value=float(params[param] * 2),
                    value=float(params[param]),
                    step=0.1 if isinstance(params[param], float) else 1.0,
                    key=f"{selected_strategy}_{param}"
                )

            # 执行策略按钮
            if st.button("执行策略优化", key="strategy_btn", use_container_width=True):
                # 模拟策略执行
                with st.spinner('策略优化中...'):
                    time.sleep(1.5)

                    # 模拟负荷和电价数据
                    load_data = df['load'].tail(96).values
                    price_data = df['electricity_price'].tail(96).values

                    # 模拟策略动作
                    action_data = ["待机"] * len(load_data)

                    if selected_strategy == "峰谷套利":
                        # 在低谷时段充电，高峰时段放电
                        min_price_idx = np.argmin(price_data)
                        max_price_idx = np.argmax(price_data)

                        if min_price_idx < max_price_idx:
                            action_data[min_price_idx] = "充电"
                            action_data[max_price_idx] = "放电"

                    elif selected_strategy == "需量控制":
                        # 在负荷接近阈值时放电
                        threshold = params["需量阈值"]
                        for i in range(len(load_data)):
                            if load_data[i] > threshold * 0.9:
                                action_data[i] = "放电"

                    # 计算策略收益
                    revenue = strategy_controller.calculate_revenue(
                        selected_strategy, params, load_data, price_data)

                    # 保存策略结果
                    st.session_state.strategy_result = {
                        "strategy": selected_strategy,
                        "params": params,
                        "load_data": load_data,
                        "price_data": price_data,
                        "action_data": action_data,
                        "revenue": revenue
                    }

        with col2:
            if "strategy_result" in st.session_state:
                result = st.session_state.strategy_result

                # 显示策略执行结果
                st.markdown(f"""
                <div class="card strategy-pulse slide-in-right">
                    <h4>策略执行结果: {result['strategy']}</h4>
                    <div class="divider"></div>
                    <div class="metric-card">
                        <div class="metric-label">预计每日收益</div>
                        <div class="metric-value" style="color: #9b5de5;">{result['revenue']:.2f} 元</div>
                    </div>
                    <div class="divider"></div>
                    <p><strong>策略动作统计:</strong></p>
                    <ul>
                        <li>充电次数: {result['action_data'].count('充电')}</li>
                        <li>放电次数: {result['action_data'].count('放电')}</li>
                        <li>待机次数: {result['action_data'].count('待机')}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # 绘制策略执行图
                fig = create_strategy_chart(
                    result['load_data'],
                    result['price_data'],
                    result['strategy'],
                    result['action_data']
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.session_state.current_tab = 4
        st.subheader("经济收益分析")
        st.markdown('<div class="financial-section">', unsafe_allow_html=True)

        # 关键指标卡片
        col1, col2, col3 = st.columns(3)
        col1.markdown("""
        <div class="metric-card energy-wave slide-in-left financial-metric">
            <div class="metric-label">年度总收益</div>
            <div class="metric-value">¥86,450</div>
            <div class="metric-label">+15.3% 同比增长</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown("""
        <div class="metric-card energy-wave slide-in-left financial-metric" style="animation-delay: 0.2s;">
            <div class="metric-label">投资回报率</div>
            <div class="metric-value">32.7%</div>
            <div class="metric-label">行业领先水平</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown("""
        <div class="metric-card energy-wave slide-in-left financial-metric" style="animation-delay: 0.4s;">
            <div class="metric-label">回收期</div>
            <div class="metric-value">3.2年</div>
            <div class="metric-label">优于预期目标</div>
        </div>
        """, unsafe_allow_html=True)

        if "strategy_result" in st.session_state:
            result = st.session_state.strategy_result

            # 生成财务报告
            financial_report = economic_analysis.generate_financial_report(
                result['strategy'], result['revenue'], result['params'])

            col1, col2 = st.columns([1, 2])
            with col1:
                # 准备指标数据
                strategy = financial_report["strategy"]
                investment = f"{financial_report['investment']:,.2f} 元"
                annual_savings = f"{financial_report['annual_savings']:,.2f} 元"
                roi = f"{financial_report['roi']:.1f}%" if financial_report['roi'] != float('inf') else "∞%"
                payback = f"{financial_report['payback_years']:.1f}年" if financial_report['payback_years'] != float(
                    'inf') else "∞年"

                # 构建HTML内容
                # st.markdown(f"""
                # <div class="card revenue-pulse slide-in-left revenue-card">
                #     <h4>{strategy}财务分析</h4>
                #
                #     <div class="metric-card financial-metric">
                #         <div class="metric-label">初始投资</div>
                #         <div class="metric-value">{investment}</div>
                #     </div>
                #
                #     <div class="metric-card financial-metric">
                #         <div class="metric-label">预计年收益</div>
                #         <div class="metric-value">{annual_savings}</div>
                #     </div>
                #
                #     <div class="divider"></div>
                #
                #     <div class="revenue-highlight">投资回报率 (ROI): {roi}</div>
                #     <div class="revenue-highlight">投资回收期: {payback}</div>
                #
                #     <div class="revenue-breakdown">
                #         <h5>收益来源分析</h5>
                #         <p><strong>峰谷套利收益:</strong> 65%</p>
                #         <p><strong>需量电费节省:</strong> 20%</p>
                #         <p><strong>需求响应补贴:</strong> 10%</p>
                #         <p><strong>光伏发电节省:</strong> 5%</p>
                #     </div>
                # </div>
                # """, unsafe_allow_html=True)
                #

                html_content = (
                    '<div class="card revenue-pulse slide-in-left">'
                    f'<h4>{strategy}财务分析</h4>'
                    '<div class="divider"></div>'

                    '<div class="metric-card">'
                    '<div class="metric-label">初始投资</div>'
                    f'<div class="metric-value" style="color: #9b5de5;">{investment}</div>'
                    '</div>'

                    '<div class="metric-card">'
                    '<div class="metric-label">预计年收益</div>'
                    f'<div class="metric-value" style="color: #2ec4b6;">{annual_savings}</div>'
                    '</div>'

                    '<div class="divider"></div>'

                    '<div class="metric-card">'
                    '<div class="metric-label">投资回报率 (ROI)</div>'
                    f'<div class="metric-value" style="color: #f9c74f;">{roi}</div>'
                    '</div>'

                    '<div class="metric-card">'
                    '<div class="metric-label">投资回收期</div>'
                    f'<div class="metric-value" style="color: #ff6b6b;">{payback}</div>'
                    '</div>'
                    '</div>'  # 关闭主卡片
                )
                # 一次性渲染整个卡片
                st.markdown(html_content, unsafe_allow_html=True)

                # 额外信息
                st.markdown("""
                <div class="card slide-in-left" style="margin-top: 1.5rem;">
                    <h4>财务分析说明</h4>
                    <div class="divider"></div>
                    <div class="revenue-detail">
                        以上分析基于当前策略参数和市场条件，实际收益可能因以下因素变化：
                    </div>
                    <ul style="font-size: 1.1rem; line-height: 1.6;">
                        <li>电价波动幅度</li>
                        <li>设备维护成本</li>
                        <li>政策补贴变化</li>
                        <li>能源市场供需关系</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # 绘制财务分析图
                fig = create_financial_chart(financial_report)
                st.plotly_chart(fig, use_container_width=True)

                # 年收益预测
                years = 5
                investment = financial_report['investment']
                savings = [0] * years
                cumulative = [0] * years

                for i in range(years):
                    savings[i] = financial_report['annual_savings'] * (1 + 0.05) ** i  # 假设每年增长5%
                    cumulative[i] = savings[i] + (cumulative[i - 1] if i > 0 else 0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(1, years + 1)),
                    y=savings,
                    name='年收益',
                    marker_color='#2ec4b6'
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, years + 1)),
                    y=cumulative,
                    mode='lines+markers',
                    name='累计收益',
                    line=dict(color='#f9c74f', width=3),
                    yaxis='y2'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, years + 1],
                    y=[investment, investment],
                    mode='lines',
                    name='初始投资',
                    line=dict(color='#ff6b6b', width=2, dash='dash')
                ))

                fig.update_layout(
                    title=dict(text="5年收益预测", font=dict(size=22, color='#2ec4b6')),
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title='年份', tickvals=list(range(1, years + 1))),
                    yaxis=dict(title='年收益 (元)'),
                    yaxis2=dict(title='累计收益 (元)', overlaying='y', side='right'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请先在'策略控制'选项卡中执行策略优化，然后查看经济收益分析")

        st.markdown('</div>', unsafe_allow_html=True)

    # 页脚
    st.markdown("""
    <div class="footer">
        <p>NeuroGrid Pro 智能能源管理平台 v3.0 | 神经网格技术驱动 | © 2023 未来能源实验室</p>
        <p>警告：本系统预测结果仅供参考，实际决策请结合专业分析</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()