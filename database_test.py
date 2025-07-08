import pandas as pd
from database.energy_database import EnergyDatabase
from datetime import datetime


# 创建数据库实例
with EnergyDatabase("energy_management.db") as db:
    # 添加用户
    user_id = db.users.add_user(
        username="carbon_user",
        password="SecurePass123!",
        company_name="Green Tech Ltd.",
        industry_type="Manufacturing"
    )

    # 认证用户
    auth_result = db.users.authenticate_user("carbon_user", "SecurePass123!")
    if auth_result:
        print(f"Authenticated user: {auth_result[0]}")

    # 创建历史数据 - 使用Python datetime对象而不是Pandas Timestamp
    timestamps = [
        datetime(2023, 1, 1, 0),
        datetime(2023, 1, 1, 1),
        datetime(2023, 1, 1, 2),
        datetime(2023, 1, 1, 3),
        datetime(2023, 1, 1, 4)
    ]

    hist_data = pd.DataFrame({
        'timestamp': timestamps,
        'load': [150.2, 145.8, 160.3, 155.7, 148.9],
        'temperature': [24.5, 23.8, 25.2, 24.9, 23.5],
        'humidity': [50.1, 48.7, 52.3, 49.8, 47.5],
        'electricity_price': [0.15, 0.14, 0.16, 0.15, 0.14]
    })

    db.historical.save_data(user_id, hist_data)
    print("Historical data saved successfully")

    # 创建预测数据 - 使用Python datetime对象
    prediction_timestamps = [
        datetime(2023, 1, 2, 0),
        datetime(2023, 1, 2, 1),
        datetime(2023, 1, 2, 2)
    ]

    # 负荷数据预测
    prediction_data = pd.DataFrame({
        'timestamp': prediction_timestamps,
        'value': [140.2, 138.7, 143.5]
    })

    db.load_predictions.save_prediction(user_id, "load_forecast", prediction_data)
    print("Load prediction data saved successfully")

    price_prediction_data = pd.DataFrame({
        'timestamp': prediction_timestamps,
        'value': [0.11, 0.138, 0.143]
    })

    db.price_predictions.save_prediction(user_id, "price_forecast", prediction_data)
    print("Price prediction data saved successfully")

    # 保存碳排放数据
    carbon_data = {
        'timestamp': datetime(2023, 1, 1, 12, 0, 0),
        'scope1': 120.5,
        'scope2': 85.3,
        'scope3': 45.2,
        'total': 251.0
    }
    db.carbon.save_emission_data(user_id, carbon_data)
    print("Carbon emission data saved successfully")

    # 保存碳平台配置
    db.carbon.save_config("carbon_target", "1000", "Annual carbon reduction target (tons)")
    db.carbon.save_config("report_interval", "quarterly", "Carbon reporting interval")
    print("Carbon config saved successfully")

    # 记录用户活动
    db.carbon.log_activity(user_id, "data_upload", "Uploaded energy consumption data")
    print("Activity logged successfully")

    # 加载碳排放数据
    emissions = db.carbon.load_emission_data(user_id)
    print("\nCarbon emission data:")
    for emission in emissions:
        print(emission)

    # 保存策略
    strategy_params = {"type": "investment"}
    strategy_id = db.strategies.save_strategy(
        user_id, "Investment Strategy", strategy_params)

    # 保存财务分析（自动加密）
    financial_data = {
        "investment": 500000,
        "annual_savings": 125000,
        "roi": 25.0,
        "payback_years": 4.0
    }
    db.financial.save_analysis(user_id, strategy_id, financial_data)
    print("Financial analysis saved (encrypted)")

    # 加载财务分析（自动解密）
    analysis = db.financial.load_analysis(user_id, strategy_id)
    if analysis:
        print("\nDecrypted financial analysis:")
        print(f"Investment: ${analysis['investment']:,.2f}")
        print(f"Annual Savings: ${analysis['annual_savings']:,.2f}")
        print(f"ROI: {analysis['roi']:.2f}%")
        print(f"Payback Years: {analysis['payback_years']:.1f}")
        print(f"Created at: {analysis['created_at']}")
    else:
        print("Failed to load financial analysis")

    # 删除用户测试
    db.users.delete_user(user_id)
    print(f"\nUser {user_id} and all related data deleted")
