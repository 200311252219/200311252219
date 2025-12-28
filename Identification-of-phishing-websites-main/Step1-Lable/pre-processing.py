import pandas as pd
import requests
import logging
from urllib.parse import urlparse
from pathlib import Path
import random
import json

# ==================================================
# 日志配置
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==================================================
# 绝对路径配置（按你现在的工程）
# ==================================================
PHISH_FILE = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step1-Lable\phish_url.csv"
OUTPUT_FILE = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step1-Lable\URL_label.csv"
COMMONCRAWL_CACHE = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step1-Lable\commoncrawl_cache.json"

# ==================================================
# 常量
# ==================================================
PHISH_LABEL = 1
LEGIT_LABEL = 0

VALID_TLDS = {
    "com", "org", "net", "edu", "gov", "info", "biz", "io"
}

SUSPICIOUS_KEYWORDS = {
    "login", "verify", "account", "update",
    "secure", "bank", "paypal", "apple", "google"
}

REQUEST_TIMEOUT = 15


# ==================================================
def load_phishing_data(file_path: str) -> pd.DataFrame:
    """加载并清洗 PhishTank 数据"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Phish file not found: {file_path}")

    df = pd.read_csv(file_path)

    if "verified" in df.columns:
        df = df[df["verified"] == "yes"]

    df = df[["url"]].dropna().drop_duplicates()
    df["label"] = PHISH_LABEL

    logging.info(f"Loaded {len(df)} verified phishing URLs")
    return df


# ==================================================
def fetch_commoncrawl_urls(num_urls: int) -> pd.DataFrame:
    """从 CommonCrawl 获取 URL（带缓存）"""
    cache_path = Path(COMMONCRAWL_CACHE)

    if cache_path.exists():
        logging.info("Loading CommonCrawl URLs from cache")
        urls = json.loads(cache_path.read_text())
    else:
        logging.info("Fetching CommonCrawl index info")
        index_api = "https://index.commoncrawl.org/collinfo.json"
        resp = requests.get(index_api, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        latest_crawl = resp.json()[0]["id"]

        search_api = (
            f"https://index.commoncrawl.org/"
            f"{latest_crawl}-index"
            f"?url=*.com&output=json"
        )

        logging.info("Fetching CommonCrawl URL records")
        resp = requests.get(search_api, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        urls = []
        for line in resp.text.splitlines():
            try:
                record = json.loads(line)
                urls.append(record["url"])
            except Exception:
                continue

        urls = list(set(urls))
        cache_path.write_text(json.dumps(urls))
        logging.info(f"Cached {len(urls)} CommonCrawl URLs")

    random.shuffle(urls)
    urls = urls[:num_urls]

    return pd.DataFrame({
        "url": urls,
        "label": LEGIT_LABEL
    })


# ==================================================
def is_suspicious_url(url: str) -> bool:
    """弱过滤：去除疑似钓鱼关键词"""
    u = url.lower()
    return any(k in u for k in SUSPICIOUS_KEYWORDS)


# ==================================================
def filter_legit_urls(df: pd.DataFrame) -> pd.DataFrame:
    """合法 URL 过滤（论文级弱过滤）"""

    def is_valid(url: str) -> bool:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            tld = parsed.netloc.split(".")[-1].lower()
            return tld in VALID_TLDS
        except Exception:
            return False

    before = len(df)

    df = df[df["url"].apply(is_valid)]
    df = df[~df["url"].apply(is_suspicious_url)]
    df = df.drop_duplicates(subset=["url"])

    logging.info(f"Filtered legitimate URLs: {before} → {len(df)}")
    return df


# ==================================================
def balance_dataset(df_phish: pd.DataFrame, df_legit: pd.DataFrame) -> pd.DataFrame:
    """平衡正负样本"""
    n = min(len(df_phish), len(df_legit))

    df = pd.concat([
        df_phish.sample(n, random_state=42),
        df_legit.sample(n, random_state=42)
    ])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"Balanced dataset: phishing={n}, legitimate={n}")
    return df


# ==================================================
def main():
    logging.info("===== Step1 (Paper-level) Started =====")

    df_phish = load_phishing_data(PHISH_FILE)

    df_legit = fetch_commoncrawl_urls(num_urls=len(df_phish) * 2)
    df_legit = filter_legit_urls(df_legit)

    df_all = balance_dataset(df_phish, df_legit)

    df_all.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Dataset saved to: {OUTPUT_FILE}")
    logging.info("===== Step1 Finished Successfully =====")


# ==================================================
if __name__ == "__main__":
    main()
