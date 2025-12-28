import streamlit as st
import pandas as pd
import joblib
import os
import urllib.parse
import socket
import tldextract
import datetime
import requests
import whois
import warnings
import sqlite3
from bs4 import BeautifulSoup

# 忽略 SSL 警告
warnings.filterwarnings("ignore")

# ======================================================
# 1. 路径配置
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Step3-Modeling")
DB_PATH = os.path.join(BASE_DIR, "phishing_system.db")

MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
SELECTOR_PATH = os.path.join(MODEL_DIR, "feature_selector.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# ======================================================
# 2. 界面配置 & 隐藏默认菜单
# ======================================================
st.set_page_config(
    page_title="智能钓鱼网站检测系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### 🛡️ 智能钓鱼网站检测系统 v1.0\n\n本系统基于 **随机森林 (Random Forest)** 算法构建，旨在帮助用户识别恶意钓鱼链接。"
    }
)

# --- 隐藏 Streamlit 默认风格 (Deploy 按钮等) ---
hide_streamlit_style = """
<style>
    .stDeployButton {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ======================================================
# 3. 数据库管理模块 (SQLite)
# ======================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS whitelist
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     domain
                     TEXT
                     UNIQUE,
                     added_at
                     TIMESTAMP
                     DEFAULT
                     CURRENT_TIMESTAMP
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     url
                     TEXT,
                     result
                     TEXT,
                     probability
                     REAL,
                     timestamp
                     TIMESTAMP
                     DEFAULT
                     CURRENT_TIMESTAMP
                 )''')

    # 预制白名单
    initial_whitelist = ["bilibili.com", "baidu.com", "qq.com", "google.com", "taobao.com", "jd.com"]
    for domain in initial_whitelist:
        try:
            c.execute("INSERT INTO whitelist (domain) VALUES (?)", (domain,))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()


def add_to_whitelist(domain):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO whitelist (domain) VALUES (?)", (domain,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_whitelist():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT domain as '域名', added_at as '添加时间' FROM whitelist ORDER BY added_at DESC",
                           conn)
    conn.close()
    return df


def save_history(url, result, probability):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (url, result, probability) VALUES (?, ?, ?)",
              (url, result, probability))
    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT url, result, probability, timestamp FROM history ORDER BY timestamp DESC LIMIT 50",
                           conn)
    conn.close()
    # 重命名列以便展示
    df.columns = ["检测链接", "检测结果", "置信度", "检测时间"]
    return df


if not os.path.exists(DB_PATH):
    init_db()


# ======================================================
# 4. 核心加载函数
# ======================================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)

        if os.path.exists(COLUMNS_PATH):
            feature_cols = joblib.load(COLUMNS_PATH)
        else:
            return None, None, None, None
        return model, scaler, selector, feature_cols
    except FileNotFoundError:
        return None, None, None, None


model, scaler, selector, feature_columns = load_resources()


# ======================================================
# 5. 业务逻辑函数
# ======================================================
def check_whitelist_db(url):
    try:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}".lower()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT 1 FROM whitelist WHERE domain = ?", (domain,))
        result = c.fetchone()
        conn.close()
        if result:
            return True, domain
    except:
        pass
    return False, None


# --- 特征提取 (保持逻辑不变) ---
def extract_address_bar_features(url):
    features = {}
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["protocol"] = 1 if urllib.parse.urlparse(url).scheme == "https" else 0
    try:
        domain = urllib.parse.urlparse(url).netloc
        socket.inet_aton(domain.split(":")[0])
        features["uses_ip"] = 1
    except:
        features["uses_ip"] = 0
    try:
        ext = tldextract.extract(url)
        features["num_subdomains"] = len(ext.subdomain.split(".")) if ext.subdomain else 0
    except:
        features["num_subdomains"] = 0
    return features


def extract_domain_features(url):
    features = {"domain_age_days": -1, "dns_valid": 0, "whois_info_exists": 0}
    try:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        try:
            w = whois.whois(domain)
            if w.creation_date:
                c = w.creation_date
                if isinstance(c, list): c = c[0]
                if isinstance(c, datetime.datetime):
                    features["domain_age_days"] = (datetime.datetime.now() - c).days
            features["whois_info_exists"] = 1 if w else 0
        except:
            pass
        try:
            socket.gethostbyname(domain)
            features["dns_valid"] = 1
        except:
            pass
    except:
        pass
    return features


def extract_html_features(url):
    features = {"has_iframe": -1, "has_obfuscated_js": -1}
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=5, verify=False)
        soup = BeautifulSoup(r.text, "html.parser")
        features["has_iframe"] = 1 if soup.find("iframe") else 0
        scripts = soup.find_all("script")
        for s in scripts:
            if s.string:
                txt = s.string.lower()
                if "eval(" in txt and len(txt) < 1000:
                    features["has_obfuscated_js"] = 1;
                    break
                if "\\x" in txt and len(txt) > 500:
                    features["has_obfuscated_js"] = 1;
                    break
        else:
            features["has_obfuscated_js"] = 0
    except:
        pass
    return features


def extract_features_pipeline(url, required_columns):
    f = {}
    f.update(extract_address_bar_features(url))
    f.update(extract_domain_features(url))
    f.update(extract_html_features(url))
    df = pd.DataFrame([f])
    df_aligned = pd.DataFrame(columns=required_columns)
    for col in required_columns:
        if col in df.columns:
            df_aligned.loc[0, col] = df.loc[0, col]
        else:
            df_aligned.loc[0, col] = 0
    return df_aligned


# ======================================================
# 6. 主界面逻辑 (全中文)
# ======================================================

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 管理后台")

    tab1, tab2 = st.tabs(["📝 历史记录", "🛡️ 白名单管理"])

    with tab1:
        st.caption("最近 50 条检测记录")
        if st.button("🔄 刷新记录", use_container_width=True):
            pass
        history_df = get_history()
        st.dataframe(history_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("添加信任域名")
        new_domain = st.text_input("输入域名 (如 jd.com)", key="new_domain")
        if st.button("➕ 添加至白名单", use_container_width=True):
            if new_domain:
                if add_to_whitelist(new_domain):
                    st.success(f"已成功添加: {new_domain}")
                else:
                    st.warning("添加失败：域名已存在或格式无效")

        st.divider()
        st.subheader("当前白名单列表")
        whitelist_df = get_whitelist()
        st.dataframe(whitelist_df, use_container_width=True, hide_index=True)

# --- 主内容区 ---
st.title("🛡️ 智能钓鱼网站检测系统")
st.markdown("### 基于机器学习 (Machine Learning) 的实时威胁检测平台")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    url_input = st.text_input("🔗 请输入目标 URL：", placeholder="例如：https://www.example.com")
with col2:
    st.write("")
    st.write("")
    check_btn = st.button("🚀 开始检测", type="primary", use_container_width=True)

if check_btn:
    if not url_input.strip():
        st.warning("⚠️ 请输入有效的 URL 链接")

    elif model is None:
        st.error("❌ 系统故障：未找到模型文件。请检查 Step3-Modeling 目录是否完整。")

    else:
        # --- 步骤 1: 数据库白名单检查 ---
        is_safe, domain_name = check_whitelist_db(url_input)

        if is_safe:
            st.balloons()
            st.success(f"✅ **检测结果：安全 (Safe)**")
            st.info(f"域名 `{domain_name}` 位于系统的信任白名单中，无需进行模型运算。")

            # 记录历史 (中文)
            save_history(url_input, "安全 (白名单)", 1.0)

            with st.expander("🔍 查看特征详情 (后台提取)"):
                with st.spinner("正在提取特征..."):
                    df_raw = extract_features_pipeline(url_input, feature_columns)
                    st.dataframe(df_raw)

        # --- 步骤 2: 模型预测 ---
        else:
            with st.spinner("🔍 正在进行深度检测 (特征提取 -> 智能分析)..."):
                try:
                    # 预测流程
                    X_input = extract_features_pipeline(url_input, feature_columns)
                    X_scaled = scaler.transform(X_input)
                    X_selected = selector.transform(X_scaled)

                    prediction = model.predict(X_selected)[0]
                    proba = model.predict_proba(X_selected)[0]

                    # 结果文案处理
                    if prediction == 1:
                        result_text = "钓鱼网站"
                        result_prob = proba[1]
                    else:
                        result_text = "正常网站"
                        result_prob = proba[0]

                    # 存入数据库
                    save_history(url_input, result_text, float(result_prob))

                    # 界面显示
                    st.divider()
                    if prediction == 1:
                        st.error(f"🚫 **检测结果：钓鱼网站 (Phishing)**")
                        st.metric(label="风险概率", value=f"{result_prob * 100:.2f}%", delta="高风险")
                        st.markdown(
                            "⚠️ **警告**：该网站命中多个恶意特征（如混淆代码、域名异常等），系统判定为高风险！请勿输入任何个人信息。")
                    else:
                        st.success(f"✅ **检测结果：正常网站 (Legitimate)**")
                        st.metric(label="安全概率", value=f"{result_prob * 100:.2f}%", delta="安全")

                    # 特征展示
                    with st.expander("🔬 技术分析报告"):
                        st.markdown("**模型输入特征向量 (Normalized):**")
                        # 简单的图表展示
                        st.bar_chart(pd.DataFrame(X_selected.T, columns=["特征值"]))
                        st.markdown("**原始特征数据:**")
                        st.dataframe(X_input)

                except Exception as e:
                    st.error(f"检测过程中发生错误: {e}")

st.markdown("---")
st.caption("© 2025 智能钓鱼网站检测系统 | 技术栈：Python • Scikit-Learn • Streamlit")
