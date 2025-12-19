import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split

# ==================================================
# ⚙️ 設定: フォルダパス (spatial_avg, NoStd を指定)
# ==================================================
# ※ お手元のフォルダ名に合わせて調整してください
TRAIN_FOLDER_NAME = "maps_POTTS_q2_R32_PBC_spatial_avg_NoStd/r_032.0000"
TEST_FOLDER_NAME = "maps_POTTS_q3_R32_PBC_critical_spatial_avg_NoStd/r_032.0000"

L = 64
EPOCHS = 30
BATCH_SIZE = 32
# Ising q=2 臨界点
BETA_C_ISING = 1.0 / (1.0 / np.log(1 + np.sqrt(2)))


# ==================================================
# 🛠️ データ読み込み (前回と同じ)
# ==================================================
def load_train_data(folder, margin=0.05):
    files = glob.glob(os.path.join(folder, "*.npy"))
    if not files:
        return None, None
    X, y = [], []
    for f in files:
        beta_match = re.search(r"_b([0-9.]+)_", os.path.basename(f))
        beta = (
            float(beta_match.group(1))
            if beta_match
            else float(re.search(r"beta_?([0-9.]+)", os.path.basename(f)).group(1))
        )

        if abs(beta - BETA_C_ISING) < margin:
            continue
        X.append(np.load(f))
        y.append(1 if beta > BETA_C_ISING else 0)
    return np.array(X).astype("float32"), np.array(y)


print("📂 Loading Data...")
X_train_raw, y_train = load_train_data(TRAIN_FOLDER_NAME)

# CNN用にリシェイプ (N, 64, 64, 1)
X_train_cnn = X_train_raw.reshape(-1, L, L, 1)
print(f"Data loaded: {X_train_cnn.shape}")

# ==================================================
# 🧠 モデル定義: 浅い vs 深い (どちらも BNなし)
# ==================================================


def build_shallow_cnn(input_shape):
    """
    【仮説検証用】前回の失敗を再現する「浅くて狭い」モデル
    - BatchNormalization: なし
    - 層数: 1層のみ
    - フィルタ数: 16 (少ない)
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # 1層目 (フィルタ数16)
            layers.Conv2D(
                16,
                (3, 3),
                activation="relu",
                padding="same",
            ),
            layers.MaxPooling2D((2, 2)),
            # ここで終わり！すぐにFlatten
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="Shallow_CNN",
    )
    return model


def build_deep_cnn(input_shape):
    """
    【比較用】今回成功した「深くて広い」モデル (ただしBNは抜く)
    - BatchNormalization: なし (検証のためあえて抜く)
    - 層数: 4層 (2ブロック)
    - フィルタ数: 32 -> 64
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="Deep_CNN",
    )
    return model


# ==================================================
# 🚀 実験実行
# ==================================================
# 1. 浅いモデル (再現)
print("\n🔥 Training Shallow CNN (Replicating failure)...")
model_shallow = build_shallow_cnn((L, L, 1))
model_shallow.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
hist_shallow = model_shallow.fit(
    X_train_cnn,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=0,
)
print("Done.")

# 2. 深いモデル (今回)
print("\n💧 Training Deep CNN (Replicating success)...")
model_deep = build_deep_cnn((L, L, 1))
model_deep.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist_deep = model_deep.fit(
    X_train_cnn,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=0,
)
print("Done.")

# ==================================================
# 📊 結果比較プロット
# ==================================================
plt.figure(figsize=(14, 5))

# Accuracy Comparison
plt.subplot(1, 2, 1)
plt.plot(
    hist_shallow.history["val_accuracy"],
    "o--",
    label="Shallow CNN (16 filters)",
    color="orange",
)
plt.plot(
    hist_deep.history["val_accuracy"],
    "s-",
    label="Deep CNN (32-64 filters)",
    color="blue",
)
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Loss Comparison
plt.subplot(1, 2, 2)
plt.plot(hist_shallow.history["val_loss"], "o--", label="Shallow CNN", color="orange")
plt.plot(hist_deep.history["val_loss"], "s-", label="Deep CNN", color="blue")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiment_shallow_vs_deep_noBN.png")
print("\n✅ Comparison graph saved as: experiment_shallow_vs_deep_noBN.png")
plt.show()
