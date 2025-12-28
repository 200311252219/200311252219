import pandas as pd
import time
import urllib.parse
import tldextract
import socket
import datetime
import requests
import joblib
import whois
import os
import warnings
import concurrent.futures
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# 1. 全局配置
# ------------------------------
warnings.filterwarnings("ignore")

# --- 核心开关 ---
ENABLE_WHOIS = True
MAX_WORKERS = 10  # 线程数

# 路径配置
INPUT_DIR = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step2-Feature"
OUTPUT_DIR = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step3-Modeling"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 文件路径
INPUT_FILE = os.path.join(INPUT_DIR, "url_features.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "url_model_final.csv")

# 模型相关文件路径
MODEL_PATH = os.path.join(OUTPUT_DIR, "phishing_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
SELECTOR_PATH = os.path.join(OUTPUT_DIR, "feature_selector.pkl")
# 【新增】列名文件，GUI 必须用
FEATURE_COLUMNS_PATH = os.path.join(OUTPUT_DIR, "feature_columns.pkl")


# ------------------------------
# 2. 特征提取函数群 (保持不变)
# ------------------------------
def extract_address_bar_features(url):
    features = {}
    features['url_length'] = len(url)
    try:
        domain = urllib.parse.urlparse(url).netloc
        if domain:
            try:
                socket.inet_aton(domain.split(':')[0])
                features['uses_ip'] = 1
            except:
                features['uses_ip'] = 0
        else:
            features['uses_ip'] = 0
    except:
        features['uses_ip'] = 0
    features['num_dots'] = url.count('.')
    features['protocol'] = 1 if urllib.parse.urlparse(url).scheme == 'https' else 0
    try:
        extracted = tldextract.extract(url)
        subdomains = extracted.subdomain.split('.')
        features['num_subdomains'] = len(subdomains) if extracted.subdomain else 0
    except:
        features['num_subdomains'] = 0
    return features


def extract_domain_based_features(url):
    features = {'domain_age_days': -1, 'dns_valid': 0, 'whois_info_exists': 0}
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        if ENABLE_WHOIS:
            try:
                w = whois.whois(domain)
                if w.creation_date:
                    c_date = w.creation_date
                    if isinstance(c_date, list): c_date = c_date[0]
                    if isinstance(c_date, datetime.datetime):
                        features['domain_age_days'] = (datetime.datetime.now() - c_date).days
                features['whois_info_exists'] = 1 if w else 0
            except:
                pass
        try:
            socket.gethostbyname(domain)
            features['dns_valid'] = 1
        except:
            pass
    except:
        pass
    return features


def extract_html_js_features(url):
    features = {'has_iframe': -1, 'has_obfuscated_js': -1}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        features['has_iframe'] = 1 if soup.find('iframe') else 0
        scripts = soup.find_all('script')
        obfuscated = 0
        for script in scripts:
            if script.string:
                content = script.string.lower()
                if 'eval(' in content or '\\x' in content or 'unescape(' in content:
                    obfuscated = 1;
                    break
        features['has_obfuscated_js'] = obfuscated
    except:
        pass
    return features


# ------------------------------
# 3. 多线程处理逻辑
# ------------------------------
def process_single_row(args):
    index, row, url_column, label_column = args
    url = row[url_column]
    features = {}
    features.update(extract_address_bar_features(url))
    features.update(extract_domain_based_features(url))
    features.update(extract_html_js_features(url))
    if label_column and label_column in row:
        features['label'] = row[label_column]
    if index % 10 == 0:
        print(f"正在处理: {url} (Index: {index})")
    return features


def process_csv_optimized(input_csv, output_csv, url_column='url', label_column=None):
    if not os.path.exists(input_csv):
        print(f"错误: 找不到文件 {input_csv}")
        return False
    print("读取 CSV 文件中...")
    df = pd.read_csv(input_csv)
    task_args = [(idx, row, url_column, label_column) for idx, row in df.iterrows()]
    results = []
    print(f"开始多线程处理 {len(df)} 条数据 (Max Workers: {MAX_WORKERS})...")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(process_single_row, task_args):
            results.append(result)
    print(f"\n特征提取结束! 耗时: {time.time() - start_time:.2f}s")
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"数据已保存: {output_csv}")
    return True


# ------------------------------
# 4. 主程序：训练与保存管道 (已优化)
# ------------------------------

if __name__ == "__main__":
    # --- 第一步：特征提取 ---
    # 如果已有 url_model_final.csv 且不想重新跑，注释下面这行
    process_success = process_csv_optimized(INPUT_FILE, OUTPUT_FILE, url_column='url', label_column='label')

    # 如果手动跳过，请取消注释下面这行
    # process_success = True

    if process_success:
        print("\n------------------------------")
        print("开始训练与保存管道 (Pipeline)")
        print("------------------------------")

        # 1. 加载数据
        data = pd.read_csv(OUTPUT_FILE)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # ==========================================
        # 【重要修复1】保存特征列名 (防止 GUI 报错)
        # ==========================================
        print(f"1. 保存特征列名到 {FEATURE_COLUMNS_PATH} ...")
        joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)

        # 2. 归一化 (Scaler)
        print("2. 训练并保存 Scaler (MinMaxScaler)...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)

        X_df = pd.DataFrame(X_scaled, columns=X.columns)

        # 3. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # ==========================================
        # 【重要修复2】特征选择策略优化 (k='all')
        # 之前的实验证明：保留所有特征(10个)比选8个准确率高 5%
        # ==========================================
        print("3. 训练并保存 Selector (Chi2)...")

        # 使用 'all' 保留所有特征，但依然保留 SelectKBest 的结构，
        # 这样可以兼容 GUI 代码 (GUI 依然期待一个 selector 对象)
        k = 'all'
        selector = SelectKBest(chi2, k=k)

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        joblib.dump(selector, SELECTOR_PATH)

        print(f"   -> 当前保留特征数: {X_train_selected.shape[1]} (已优化为全特征)")

        # 5. 模型训练
        print("4. 训练并保存 模型 (RandomForest)...")
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X_train_selected, y_train)
        joblib.dump(model, MODEL_PATH)

        # 6. 评估
        print("\n------------------------------")
        print("最终评估")
        y_pred = model.predict(X_test_selected)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # 7. 检查
        print("\n[系统自检] 必须包含以下 4 个 .pkl 文件:")
        files = [MODEL_PATH, SCALER_PATH, SELECTOR_PATH, FEATURE_COLUMNS_PATH]
        for f in files:
            exists = "✅" if os.path.exists(f) else "❌"
            print(f"{exists} {os.path.basename(f)}")