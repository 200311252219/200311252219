import pandas as pd
import joblib
import os

# ================= 配置路径 =================
# 必须指向你存放 url_model_final.csv 的文件夹
BASE_DIR = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step3-Modeling"

# 输入：你已经生成的特征数据文件
CSV_PATH = os.path.join(BASE_DIR, "url_model_final.csv")

# 输出：GUI 缺少的列名文件
OUTPUT_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")


# ================= 执行逻辑 =================
def fix_columns():
    print(f"正在检查数据文件: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print("❌ 错误: 找不到 csv 文件！请确认你已经运行过特征提取和模型训练。")
        return

    # 读取数据
    df = pd.read_csv(CSV_PATH)

    # 获取特征列名
    # 逻辑必须和训练时一致：取除了最后一列(label)之外的所有列名
    feature_cols = df.columns[:-1].tolist()

    print(f"✅ 成功提取到 {len(feature_cols)} 个特征列名。")
    print(f"示例: {feature_cols[:5]} ...")

    # 保存为 pkl
    joblib.dump(feature_cols, OUTPUT_PATH)
    print(f"✅ 文件已保存至: {OUTPUT_PATH}")
    print("现在你可以去运行 GUI 了！")


if __name__ == "__main__":
    fix_columns()