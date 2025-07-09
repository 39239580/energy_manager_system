import streamlit as st
from database.energy_database import EnergyDatabase
from pages_functionality.components.css_utils import set_custom_css
from pages_functionality.components.parts import navigation_bar, navigation_bar_info


def MainPage():
    db = EnergyDatabase()
    set_custom_css()
    # 页面配置
    st.set_page_config(
        page_title="AI Powered Energy Management Platform - NeuroGrid Pro - 智能能源管理平台",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # 用户认证
    session_state = st.session_state
    if 'user_id' not in session_state:
        session_state.user_id = None
        session_state.company_name = None

    # 登录/注册界面
    if not session_state.user_id:
        st.title("NeuroGrid Pro - 智能能源管理平台")

        tab1, tab2 = st.tabs(["登录", "注册"])

        with tab1:
            with st.form("登录表单"):
                username = st.text_input("用户名")
                password = st.text_input("密码", type="password")
                submit = st.form_submit_button("登录")

                if submit:
                    user_data = db.users.authenticate_user(username, password)
                    if user_data:
                        session_state.user_id = user_data[0]
                        session_state.company_name = user_data[1]
                        st.success(f"欢迎回来，{user_data[1]}！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

        with tab2:
            with st.form("注册表单"):
                new_username = st.text_input("用户名")
                new_password = st.text_input("密码", type="password")
                confirm_password = st.text_input("确认密码", type="password")
                company_name = st.text_input("公司名称")
                industry_type = st.selectbox("行业类型", [
                    "制造业", "商业建筑", "数据中心", "医疗", "教育", "其他"
                ])
                submit = st.form_submit_button("注册")

                if submit:
                    if new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                    else:
                        user_id = db.users.add_user(new_username, new_password, company_name, industry_type)
                        session_state.user_id = user_id
                        session_state.company_name = company_name
                        st.success("注册成功！")
                        st.rerun()

        st.stop()

    with st.container():
        # 主界面
        navigation_bar_info(session_state)
    tab1, tab2, tab3, tab4, tab5, tab6 = navigation_bar()



if __name__ == '__main__':
    MainPage()