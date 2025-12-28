import pandas as pd
import numpy as np
import joblib  # 用于保存 scaler
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ===============================
# 1. Load dataset
# ===============================
file_path = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step3-Modeling\url_model_final.csv"
data = pd.read_csv(file_path)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset loaded. Train: {X_train.shape}, Test: {X_test.shape}")

# ===============================
# 2. Random Forest (Baseline & Feature Selection)
# ===============================
print("\n[Step 2] Training Random Forest Baseline...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("-" * 30)
print(f"RF Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print("-" * 30)

# --- 特征筛选 ---
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
selected_features = importance_df[importance_df["Importance"] > 0.01]["Feature"].tolist()

print(f"Feature Selection: Retained {len(selected_features)} features out of {X.shape[1]}")

# 应用特征筛选
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# ===============================
# 3. Standardization (Crucial for Neural Nets)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# ===============================
# 4. Prepare PyTorch DataLoader
# ===============================
# 转换为 Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 使用 TensorDataset 和 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ===============================
# 5. Define Neural Network
# ===============================
class FCNet(nn.Module):
    def __init__(self, input_dim):
        super(FCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 增加 Dropout 防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出 2 类 (logits)
        )

    def forward(self, x):
        return self.net(x)


model = FCNet(input_dim=len(selected_features))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 6. Train FC Network
# ===============================
EPOCHS = 50  # 表格数据通常不需要太多 Epoch
print(f"\n[Step 6] Training Neural Network for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# ===============================
# 7. Evaluate & Compare
# ===============================
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)

fc_acc = accuracy_score(y_test_tensor, preds)
print("\n================ FINAL COMPARISON ================")
print(f"🌲 Random Forest Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
print(f"🧠 Neural Network Accuracy: {fc_acc:.4f}")
print("==================================================")
print("\nNeural Network Detailed Report:")
print(classification_report(y_test_tensor, preds))

# ===============================
# 8. (Optional) Save the Best Model
# ===============================
# 如果你想保存 PyTorch 模型供 Streamlit 使用，取消下面注释
# save_dir = r"D:\Identification-of-phishing-websites-main\Identification-of-phishing-websites-main\Step3-Modeling"
# torch.save(model.state_dict(), os.path.join(save_dir, "phishing_model_nn.pth"))
# joblib.dump(scaler, os.path.join(save_dir, "nn_scaler.pkl"))
# joblib.dump(selected_features, os.path.join(save_dir, "nn_feature_columns.pkl"))