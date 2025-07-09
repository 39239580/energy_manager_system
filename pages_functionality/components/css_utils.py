import streamlit as st


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
