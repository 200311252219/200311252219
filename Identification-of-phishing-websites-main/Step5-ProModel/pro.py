import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ================= 1. 配置路径与风格 =================
BASE_DIR = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step3-Modeling"
DATA_PATH = os.path.join(BASE_DIR, "url_model_final.csv")
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
SELECTOR_PATH = os.path.join(BASE_DIR, "feature_selector.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 数据加载与预处理 =================
def load_and_prep_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到文件 {DATA_PATH}")
        return None, None

    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y


# ================= 3. 实验与可视化函数 =================
def plot_roc_curves(X_train, X_test, y_train, y_test):
    models = [
        ("随机森林 (RF)", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("XGBoost", XGBClassifier(eval_metric='logloss', random_state=42)),  # 修复警告
        ("LightGBM", LGBMClassifier(random_state=42, verbose=-1))
    ]

    plt.figure(figsize=(10, 8))
    for name, model in models:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('不同模型的 ROC 曲线对比')
    plt.legend(loc="lower right")

    save_path = os.path.join(BASE_DIR, "roc_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ ROC 对比图已保存: {save_path}")
    plt.show()
    return models


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(feature_names))  # 最多显示10个

    plt.figure(figsize=(10, 6))
    plt.title(f"特征重要性排序 (基于 {type(model).__name__})")
    plt.barh(range(top_n), importances[indices][:top_n], align="center", color='skyblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])
    plt.xlabel("相对重要性")
    plt.gca().invert_yaxis()

    save_path = os.path.join(BASE_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 特征重要性图已保存: {save_path}")
    plt.show()


def ablation_study(X, y):
    print("\n" + "=" * 40)
    print("🧪 正在执行消融实验 (Ablation Study)")
    print("=" * 40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)

    # 全特征
    rf.fit(X_train, y_train)
    acc_full = accuracy_score(y_test, rf.predict(X_test))
    print(f"1. 使用全部 {X.shape[1]} 个特征: Accuracy = {acc_full:.4f}")

    # 特征选择
    selector = SelectKBest(chi2, k=8)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    rf.fit(X_train_sel, y_train)
    acc_sel = accuracy_score(y_test, rf.predict(X_test_sel))
    print(f"2. 使用筛选后 8 个特征: Accuracy = {acc_sel:.4f}")

    change = (acc_sel - acc_full) * 100
    print(f"👉 实验结论: 特征选择导致性能变化 {change:+.2f}%")
    print("=" * 40)


# ================= 4. 主程序 =================
if __name__ == "__main__":
    X, y = load_and_prep_data()

    if X is not None:
        # --- 阶段一：实验与分析 ---
        print("\n--- [阶段一] 正在进行模型对比与实验分析 ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ablation_study(X, y)
        trained_models = plot_roc_curves(X_train, X_test, y_train, y_test)
        plot_feature_importance(trained_models[0][1], X.columns)

        # --- 阶段二：生成并保存最终的最佳模型 ---
        print("\n" + "=" * 50)
        print("🚀 [阶段二] 正在生成并保存最终的最佳模型 (使用全部特征)")
        print("=" * 50)

        # 1. 保存特征列名
        joblib.dump(list(X.columns), COLUMNS_PATH)
        print(f"[1/4] 特征列名已保存: {COLUMNS_PATH}")

        # 2. 训练并保存 Scaler (在全部数据上训练)
        scaler_final = MinMaxScaler().fit(X)
        joblib.dump(scaler_final, SCALER_PATH)
        print(f"[2/4] 归一化器 (Scaler) 已保存: {SCALER_PATH}")
        X_final_scaled = scaler_final.transform(X)

        # 3. 训练并保存 Selector (k='all' 表示保留所有特征)
        selector_final = SelectKBest(chi2, k='all').fit(X_final_scaled, y)
        joblib.dump(selector_final, SELECTOR_PATH)
        print(f"[3/4] 特征选择器 (Selector) 已保存 (k=all): {SELECTOR_PATH}")
        X_final_selected = selector_final.transform(X_final_scaled)

        # 4. 训练并保存最终模型 (Random Forest)
        model_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_final.fit(X_final_selected, y)
        joblib.dump(model_final, MODEL_PATH)
        print(f"[4/4] 最终模型 (Random Forest) 已保存: {MODEL_PATH}")

        print("\n✨ 所有模型文件已生成完毕，GUI 系统现在可以使用最高性能的模型了！")
