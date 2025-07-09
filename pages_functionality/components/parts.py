
# 导入numpy库
import numpy as np
import streamlit as st


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


def navigation_bar_info(session_state):  # 导航栏背景设置
    # 主界面
    st.title(f"NeuroGrid Pro - {session_state.company_name} 智能能源管理平台")
    #             <div style="border-left: 4px solid #00c7c7; padding-left: 1rem; margin-bottom: 2rem;">
    st.markdown("""

            <div class="dashboard-header fade-in">
                <p>基于深度学习的能源管理系统，集负荷预测、电价预测、运行策略优化和经济收益, 能碳优化分析于一体。</p>
                <p>通过神经网格技术优化能源分配，最大化工商业用户的经济效益。</p>
            </div>
            """, unsafe_allow_html=True)
    # 创建选项卡


def navigation_bar():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 数据探索", "🔌 电价预测", "🔮 负荷预测", "⚙️ 策略控制", "💰 经济收益",
                                                  "🌱 碳管理"])
    return tab1, tab2, tab3, tab4, tab5, tab6

