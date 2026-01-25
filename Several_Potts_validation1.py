import numpy as np
import os
import glob
import re
import random
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==================================================
# ⚙️ 実験設定
# ==================================================
# 検証したいシード値のリスト
SEEDS = [0, 42, 123, 456]

# 学習用: q=2 (Ising)
TRAIN_FOLDER_NAME = "maps_POTTS_q2_R32_PBC_spatial_map_NoStd/r_032.0000"
# テスト用: q=3 (Potts)
TEST_FOLDER_NAME = "maps_POTTS_q3_R32_PBC_spatial_map_NoStd/r_032.0000"

# システムサイズ
L = 64

# Isingモデル(q=2)の臨界温度
T_C_ISING = 1.0 / np.log(1 + np.sqrt(2))
BETA_C_ISING = 1.0 / T_C_ISING

# Pottsモデル(q=3)の臨界温度 (テストデータの正解ラベル生成用)
T_C_POTTS = 1.0 / np.log(1 + np.sqrt(3))
BETA_C_POTTS = 1.0 / T_C_POTTS

# 訓練パラメータ
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
MARGIN = 0.05  # 臨界点周辺の除外マージン

print(f"📊 Configuration:")
print(f"   Ising Tc = {T_C_ISING:.4f}, Target Potts Tc = {T_C_POTTS:.4f}")
print(f"   Seeds to test: {SEEDS}")


# ==================================================
# 🛠️ ヘルパー関数
# ==================================================
def set_seed(seed):
    """乱数シードを固定して再現性を確保"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 決定論的動作を強制する場合（速度低下の可能性あり）
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'


def extract_beta(filename):
    match = re.search(r"_b([0-9.]+)_", filename)
    if match:
        return float(match.group(1))
    match = re.search(r"beta_?([0-9.]+)", filename)
    return float(match.group(1)) if match else None


def load_train_data(folder, margin=MARGIN):
    if not os.path.exists(folder):
        return None, None
    files = glob.glob(os.path.join(folder, "*.npy"))
    X, y = [], []
    for f in files:
        beta = extract_beta(os.path.basename(f))
        if beta is None:
            continue
        # 転移点付近を除外
        if abs(beta - BETA_C_ISING) < margin:
            continue
        X.append(np.load(f).astype("float32"))
        y.append(1 if beta > BETA_C_ISING else 0)  # 1:Low-T(Ordered), 0:High-T
    return np.array(X), np.array(y)


def load_test_data_with_labels(folder):
    """テストデータを読み込み、理論上のTcに基づいて正解ラベルを生成"""
    if not os.path.exists(folder):
        return None, None, None
    files = glob.glob(os.path.join(folder, "*.npy"))
    X, y, betas = [], [], []
    for f in files:
        beta = extract_beta(os.path.basename(f))
        if beta is None:
            continue
        X.append(np.load(f).astype("float32"))
        betas.append(beta)
        # Pottsモデルの正解ラベル: beta > beta_c_potts -> Ordered(1)
        y.append(1 if beta > BETA_C_POTTS else 0)
    return np.array(X), np.array(y), np.array(betas)


# ==================================================
# 🧠 モデル定義
# ==================================================
def build_fcn_model(input_dim):
    return models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                100, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def build_cnn_model(input_shape):
    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            layers.BatchNormalization(),  # ★BatchNormあり
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Block 2
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(0.001),
            ),
            layers.BatchNormalization(),  # ★BatchNormあり
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Output
            layers.Flatten(),
            layers.Dense(
                128, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


# ==================================================
# 🚀 メイン処理: マルチシード実験
# ==================================================
if __name__ == "__main__":
    # 1. データ読み込み（1回だけ実行）
    print("\n📂 Loading Data...")
    X_train_raw, y_train = load_train_data(TRAIN_FOLDER_NAME)
    X_test_raw, y_test, test_betas = load_test_data_with_labels(TEST_FOLDER_NAME)

    if X_train_raw is None or X_test_raw is None:
        print("❌ Error: Data load failed.")
        exit(1)

    print(f"   Train samples: {len(X_train_raw)}")
    print(f"   Test samples:  {len(X_test_raw)}")

    # 2. データ前処理
    # CNN用 (Reshapeのみ)
    X_train_cnn = X_train_raw.reshape(-1, L, L, 1)
    X_test_cnn = X_test_raw.reshape(-1, L, L, 1)

    # FCN用 (Flatten + Standardize)
    X_train_fcn = X_train_raw.reshape(-1, L * L)
    X_test_fcn = X_test_raw.reshape(-1, L * L)
    scaler = StandardScaler()
    X_train_fcn = scaler.fit_transform(X_train_fcn)
    X_test_fcn = scaler.transform(X_test_fcn)

    # 結果保存用リスト
    results = []

    print("\n" + "=" * 70)
    print("🚀 Starting Multi-Seed Experiment")
    print("=" * 70)

    # 3. シードごとのループ
    for seed in SEEDS:
        print(f"\n🌱 Testing Seed: {seed}")
        set_seed(seed)  # ★ここでシード固定

        # --- FCN Training ---
        print("   [FCN] Training...")
        fcn_model = build_fcn_model(L * L)
        fcn_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        fcn_model.fit(
            X_train_fcn,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[es],
            verbose=0,
        )

        # Evaluate FCN
        y_pred_fcn_prob = fcn_model.predict(X_test_fcn, verbose=0).flatten()
        y_pred_fcn = (y_pred_fcn_prob > 0.5).astype(int)
        acc_fcn = accuracy_score(y_test, y_pred_fcn)

        # --- CNN Training ---
        print("   [CNN] Training...")
        # モデル再構築（初期化）
        cnn_model = build_cnn_model((L, L, 1))
        cnn_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        cnn_model.fit(
            X_train_cnn,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[es],
            verbose=0,
        )

        # Evaluate CNN
        y_pred_cnn_prob = cnn_model.predict(X_test_cnn, verbose=0).flatten()
        y_pred_cnn = (y_pred_cnn_prob > 0.5).astype(int)
        acc_cnn = accuracy_score(y_test, y_pred_cnn)

        print(f"   -> Result: FCN Acc={acc_fcn:.4f}, CNN Acc={acc_cnn:.4f}")
        results.append({"seed": seed, "fcn_acc": acc_fcn, "cnn_acc": acc_cnn})

        # メモリクリア
        tf.keras.backend.clear_session()

    # ==================================================
    # 📊 集計と出力
    # ==================================================
    print("\n" + "=" * 70)
    print("📈 Experiment Summary")
    print("=" * 70)
    print(f"{'Seed':<10} | {'FCN Accuracy':<15} | {'CNN Accuracy':<15}")
    print("-" * 46)

    fcn_accs = []
    cnn_accs = []

    for res in results:
        print(
            f"{res['seed']:<10} | {res['fcn_acc']:.4f}          | {res['cnn_acc']:.4f}"
        )
        fcn_accs.append(res["fcn_acc"])
        cnn_accs.append(res["cnn_acc"])

    print("-" * 46)
    print(
        f"{'Average':<10} | {np.mean(fcn_accs):.4f} ± {np.std(fcn_accs):.4f} | {np.mean(cnn_accs):.4f} ± {np.std(cnn_accs):.4f}"
    )

    # ファイル保存
    csv_file = "seed_experiment_results_critical.csv"
    with open(csv_file, "w") as f:
        f.write("seed,fcn_accuracy,cnn_accuracy\n")
        for res in results:
            f.write(f"{res['seed']},{res['fcn_acc']},{res['cnn_acc']}\n")

    print(f"\n📝 Results saved to: {csv_file}")
