import re
import math
import urllib.parse
import tldextract
import datetime
import whois
import requests
from bs4 import BeautifulSoup
import pandas as pd
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import os

ssl._create_default_https_context = ssl._create_unverified_context

# ==================================================
# 基础配置
# ==================================================
INPUT_FILE = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step1-Lable\URL_label.csv"
OUTPUT_FILE = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step2-Feature\url_features.csv"
PROGRESS_FILE = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step2-Feature\progress.txt"

REQUEST_TIMEOUT = 8
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "account", "secure",
    "bank", "paypal", "signin", "confirm"
]

PROGRESS_SAVE_INTERVAL = 100  # 每处理 100 条保存一次进度

# ==================================================
# 工具函数
# ==================================================
def shannon_entropy(s):
    if not s:
        return 0
    counts = Counter(s)
    return -sum((c/len(s)) * math.log2(c/len(s)) for c in counts.values())

def load_progress():
    """加载进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progress = int(f.read().strip())
        print(f"恢复进度：从 {progress} 条 URL 开始")
        return progress
    return 0

def save_progress(progress):
    """保存进度"""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(progress))
    print(f"进度保存：已处理 {progress} 条 URL")

# ==================================================
# ① URL 词法特征（论文核心）
# ==================================================
def extract_lexical_features(url):
    features = {}
    parsed = urllib.parse.urlparse(url)

    features["url_length"] = len(url)
    features["host_length"] = len(parsed.netloc)
    features["path_length"] = len(parsed.path)
    features["query_length"] = len(parsed.query)

    features["num_dots"] = url.count(".")
    features["num_hyphen"] = url.count("-")
    features["num_at"] = url.count("@")
    features["num_question"] = url.count("?")
    features["num_equal"] = url.count("=")
    features["num_slash"] = url.count("/")
    features["num_percent"] = url.count("%")

    features["num_digits"] = sum(c.isdigit() for c in url)
    features["digit_ratio"] = features["num_digits"] / len(url)

    features["has_ip"] = 0
    try:
        socket.inet_aton(parsed.netloc.split(":")[0])
        features["has_ip"] = 1
    except:
        pass

    extracted = tldextract.extract(url)
    features["num_subdomains"] = len(extracted.subdomain.split(".")) if extracted.subdomain else 0

    features["is_https"] = 1 if parsed.scheme == "https" else 0

    features["url_entropy"] = shannon_entropy(url)

    for w in SUSPICIOUS_WORDS:
        features[f"has_{w}"] = 1 if w in url.lower() else 0

    return features


# ==================================================
# ② 域名 / WHOIS / DNS 特征
# ==================================================
def extract_domain_features(url):
    features = {
        "domain_age_days": -1,
        "dns_valid": 0,
        "whois_exists": 0
    }

    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"

        try:
            w = whois.whois(domain)
            if w and w.creation_date:
                cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                features["domain_age_days"] = (datetime.datetime.now() - cd).days
                features["whois_exists"] = 1
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


# ==================================================
# ③ HTML / JS 特征（并发优化部分）
# ==================================================
def extract_html_features(url):
    features = {
        "has_iframe": -1,
        "num_iframe": -1,
        "num_script": -1,
        "has_obfuscated_js": -1,
        "external_form": -1,
        "num_external_links": -1
    }

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        iframes = soup.find_all("iframe")
        scripts = soup.find_all("script")
        links = soup.find_all("a")

        features["has_iframe"] = 1 if iframes else 0
        features["num_iframe"] = len(iframes)
        features["num_script"] = len(scripts)

        obf = 0
        for s in scripts:
            if s.string and ("eval(" in s.string or "\\x" in s.string):
                obf = 1
                break
        features["has_obfuscated_js"] = obf

        forms = soup.find_all("form")
        external = 0
        for f in forms:
            action = f.get("action", "")
            if action.startswith("http"):
                external = 1
                break
        features["external_form"] = external

        ext_links = 0
        for a in links:
            href = a.get("href", "")
            if href.startswith("http") and urlparse(url).netloc not in href:
                ext_links += 1
        features["num_external_links"] = ext_links

    except:
        pass

    return features


# ==================================================
# 单条 URL 处理（集成所有特征提取）
# ==================================================
def extract_all_features(row):
    """ 提取所有特征，row 是 Series 类型 """
    url = row[1]["url"]  # row[1] 获取行数据（pandas.Series）
    label = row[1]["label"]

    features = {"url": url}
    features.update(extract_lexical_features(url))
    features.update(extract_domain_features(url))
    features.update(extract_html_features(url))
    features["label"] = label

    return features


# ==================================================
# 并发处理核心流程
# ==================================================
def process_urls_in_parallel(df, start_index):
    """跳过已经处理过的 URL，断点继续"""
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, result in enumerate(executor.map(extract_all_features, df.iloc[start_index:].iterrows())):
            results.append(result)

            # 每处理一定数量的 URL 后保存进度
            if (i + 1) % PROGRESS_SAVE_INTERVAL == 0:
                save_progress(start_index + i + 1)

    return results


# ==================================================
# 主流程
# ==================================================
def main():
    df = pd.read_csv(INPUT_FILE)
    start_index = load_progress()  # 加载之前保存的进度

    # 从上次中断的地方继续
    results = process_urls_in_parallel(df, start_index)

    # 保存结果
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # 保存进度
    save_progress(start_index + len(results))
    # ================= 修改开始 =================
    if len(results) > 0:
        print(f"\n✅ Step2 完成，本次新增处理 {len(results)} 条，特征数 ≈ {len(results[0]) - 2}")
    else:
        print("\n✅ Step2 完成。本次运行未产生新数据（可能所有数据已在之前的运行中处理完毕）。")
    # ================= 修改结束 =================

if __name__ == "__main__":
    main()
