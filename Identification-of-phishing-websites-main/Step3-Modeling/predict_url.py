import sys
import urllib.parse
import tldextract
import socket
import datetime
import requests
import whois
import joblib
import warnings
import pandas as pd
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ======================
# 模型路径（与你训练时一致）
# ======================
MODEL_PATH = "phishing_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
SELECTOR_PATH = "feature_selector.pkl"

ENABLE_WHOIS = True  # 预测阶段可关闭以加快速度


# ======================
# 特征提取函数（必须与训练一致）
# ======================
def extract_address_bar_features(url):
    features = {}
    features['url_length'] = len(url)

    try:
        domain = urllib.parse.urlparse(url).netloc
        socket.inet_aton(domain.split(':')[0])
        features['uses_ip'] = 1
    except:
        features['uses_ip'] = 0

    features['num_dots'] = url.count('.')
    features['protocol'] = 1 if urllib.parse.urlparse(url).scheme == 'https' else 0

    try:
        extracted = tldextract.extract(url)
        features['num_subdomains'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
    except:
        features['num_subdomains'] = 0

    return features


def extract_domain_based_features(url):
    features = {
        'domain_age_days': -1,
        'dns_valid': 0,
        'whois_info_exists': 0
    }

    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"

        if ENABLE_WHOIS:
            try:
                w = whois.whois(domain)
                if w.creation_date:
                    c = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                    if isinstance(c, datetime.datetime):
                        features['domain_age_days'] = (datetime.datetime.now() - c).days
                features['whois_info_exists'] = 1
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
    features = {
        'has_iframe': -1,
        'has_obfuscated_js': -1
    }

    try:
        response = requests.get(url, timeout=3, verify=False,
                                headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        features['has_iframe'] = 1 if soup.find('iframe') else 0

        scripts = soup.find_all('script')
        obfuscated = 0
        for s in scripts:
            if s.string:
                txt = s.string.lower()
                if 'eval(' in txt or '\\x' in txt or 'unescape(' in txt:
                    obfuscated = 1
                    break
        features['has_obfuscated_js'] = obfuscated
    except:
        pass

    return features


def extract_all_features(url):
    features = {}
    features.update(extract_address_bar_features(url))
    features.update(extract_domain_based_features(url))
    features.update(extract_html_js_features(url))
    return features


# ======================
# 主预测逻辑
# ======================
def predict_url(url):
    # 1. 特征提取
    features = extract_all_features(url)
    X = pd.DataFrame([features])

    # 2. 加载预处理与模型
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    model = joblib.load(MODEL_PATH)

    # 3. 预处理
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    # 4. 预测
    pred = model.predict(X_selected)[0]
    prob = model.predict_proba(X_selected)[0][pred]

    return pred, prob


# ======================
# CLI 入口
# ======================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python predict_url.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    label, confidence = predict_url(url)

    print("=" * 50)
    print(f"URL: {url}")

    if label == 1:
        print("Prediction: 🚨 Phishing Website")
    else:
        print("Prediction: ✅ Legitimate Website")

    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)
