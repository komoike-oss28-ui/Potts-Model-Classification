import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import time
import csv
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==================================================
# ⚙️ 設定: フォルダパスとパラメータ
# ==================================================
# 学習用: q=2 (Ising)
TRAIN_FOLDER_NAME = "maps_POTTS_q2_R32_PBC_fix_origin_NoStd/r_032.0000"
# テスト用: q=3 (Potts)
TEST_FOLDER_NAME = "maps_POTTS_q3_R32_critical_PBC_fix_origin_NoStd/r_032.0000"

# システムサイズ
L = 64

# Isingモデル(q=2)の臨界温度
# Tc = 1 / ln(1 + sqrt(2)) ≈ 2.269
T_C_ISING = 1.0 / np.log(1 + np.sqrt(2))
BETA_C_ISING = 1.0 / T_C_ISING

# 訓練パラメータ
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
MARGIN = 0.05  # 臨界点周辺の除外マージン

print(f"📊 Configuration:")
print(f"   Ising Tc = {T_C_ISING:.4f}, beta_c = {BETA_C_ISING:.4f}")
print(f"   Training epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Exclusion margin: ±{MARGIN} around beta_c")


# ==================================================
# ⏱️ カスタムCallback: エポック時間計測
# ==================================================
class EpochTimeCallback(callbacks.Callback):
    """各エポックの学習時間を記録するカスタムCallback"""
    
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"   ⏱️  Epoch {epoch+1} time: {epoch_time:.2f}s")
    
    def get_average_epoch_time(self):
        """平均エポック時間を返す"""
        return np.mean(self.epoch_times) if self.epoch_times else 0.0
    
    def get_total_time(self):
        """総学習時間を返す"""
        return np.sum(self.epoch_times) if self.epoch_times else 0.0


# ==================================================
# 🛠️ ヘルパー関数
# ==================================================
def extract_beta(filename):
    """
    ファイル名からbeta値を抽出
    想定形式: conf_low_b0.1234_n001.npy または conf_high_b2.3456_n001.npy
    """
    # 'b' の後の数値を抽出
    match = re.search(r"_b([0-9.]+)_", filename)
    if match:
        return float(match.group(1))

    # フォールバック: beta_X.XXX 形式も試す
    match = re.search(r"beta_?([0-9.]+)", filename)
    return float(match.group(1)) if match else None


def load_train_data(folder, margin=MARGIN):
    """
    学習データ(q=2)読み込み: 転移点付近を除外
    Returns: X (samples, L, L), y (samples,)
    """
    if not os.path.exists(folder):
        print(f"❌ Error: Folder not found: {folder}")
        return None, None, None

    files = glob.glob(os.path.join(folder, "*.npy"))
    if not files:
        print(f"❌ Error: No .npy files found in {folder}")
        return None, None, None

    X, y, betas_used = [], [], []
    skipped_transition = 0

    for f in files:
        beta = extract_beta(os.path.basename(f))
        if beta is None:
            continue

        # 転移点付近を除外
        if abs(beta - BETA_C_ISING) < margin:
            skipped_transition += 1
            continue

        data = np.load(f).astype("float32")
        X.append(data)

        # ラベル付け: beta > beta_c → 秩序相 (1), beta < beta_c → 無秩序相 (0)
        y.append(1 if beta > BETA_C_ISING else 0)
        betas_used.append(beta)

    print(f"   Loaded {len(X)} samples ({skipped_transition} near-Tc samples excluded)")

    if len(X) == 0:
        return None, None, None

    return np.array(X), np.array(y), np.array(betas_used)


def load_test_data_with_beta(folder):
    """テストデータ(q=3)読み込み: 全温度域を使用"""
    if not os.path.exists(folder):
        print(f"❌ Error: Folder not found: {folder}")
        return None, None

    files = glob.glob(os.path.join(folder, "*.npy"))
    if not files:
        print(f"❌ Error: No .npy files found in {folder}")
        return None, None

    X, betas = [], []

    for f in files:
        beta = extract_beta(os.path.basename(f))
        if beta is None:
            continue

        data = np.load(f).astype("float32")
        X.append(data)
        betas.append(beta)

    print(f"   Loaded {len(X)} test samples")
    return np.array(X), np.array(betas)


def calculate_test_accuracy(model, X_test, test_betas, beta_c, margin=0.05):
    """
    テストデータでの精度を計算
    beta_c周辺のmargin範囲を除外して評価
    """
    predictions = model.predict(X_test, verbose=0).flatten()
    
    # True labels: beta > beta_c → 1 (ordered), beta < beta_c → 0 (disordered)
    true_labels = []
    filtered_preds = []
    
    for i, beta in enumerate(test_betas):
        # 転移点付近を除外
        if abs(beta - beta_c) < margin:
            continue
        
        true_label = 1 if beta > beta_c else 0
        true_labels.append(true_label)
        filtered_preds.append(predictions[i])
    
    if len(true_labels) == 0:
        return 0.0
    
    # 予測ラベル (0.5を閾値)
    pred_labels = (np.array(filtered_preds) > 0.5).astype(int)
    
    # 精度計算
    accuracy = np.mean(pred_labels == np.array(true_labels))
    return accuracy


def save_metrics_to_csv(metrics_data, filename="training_metrics.csv"):
    """
    学習メトリクスをCSVファイルに保存
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Avg_Epoch_Time_s', 'Total_Training_Time_s', 'Test_Accuracy'])
        for row in metrics_data:
            writer.writerow(row)
    print(f"✅ Metrics saved to: {filename}")


def save_metrics_to_txt(metrics_data, filename="training_metrics.txt"):
    """
    学習メトリクスをタブ区切りテキストファイルに保存
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Model\tAvg_Epoch_Time_s\tTotal_Training_Time_s\tTest_Accuracy\n")
        for row in metrics_data:
            f.write(f"{row[0]}\t{row[1]:.4f}\t{row[2]:.4f}\t{row[3]:.4f}\n")
    print(f"✅ Metrics saved to: {filename}")


def plot_comparison_graphs(metrics_data, config_name):
    """
    計算コストと精度の比較グラフを生成
    """
    models = [row[0] for row in metrics_data]
    avg_epoch_times = [row[1] for row in metrics_data]
    total_times = [row[2] for row in metrics_data]
    accuracies = [row[3] for row in metrics_data]
    
    # カラー設定
    colors = ['#3498db', '#e74c3c']  # Blue for FCN, Red for CNN
    
    # 2つのグラフを作成
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ====== グラフA: 1エポックあたりの学習時間 ======
    ax1 = axes[0]
    bars1 = ax1.bar(models, avg_epoch_times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Average Time per Epoch (seconds)', fontsize=18, fontweight='bold')
    ax1.set_title('Training Speed Comparison\n(Time per Epoch)', fontsize=20, fontweight='bold', pad=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 値をバーの上に表示
    for bar, val in zip(bars1, avg_epoch_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(avg_epoch_times)*0.02,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # ====== グラフB: テスト精度 ======
    ax2 = axes[1]
    bars2 = ax2.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Test Accuracy', fontsize=18, fontweight='bold')
    ax2.set_title('Model Performance Comparison\n(Test Accuracy)', fontsize=20, fontweight='bold', pad=20)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 値をバーの上に表示
    for bar, val in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # 全体タイトル
    fig.suptitle(f'Model Comparison: Cost vs Performance\n{config_name}', 
                fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存
    save_path = f"Model_Comparison_{config_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison graph saved: {save_path}")
    plt.show()


# ==================================================
# 🧠 モデル定義
# ==================================================
def build_fcn_model(input_dim, hidden_units=100, l2_reg=0.001):
    """
    FCN (Fully Connected Network)
    論文ベースだが正則化を追加
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                hidden_units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def build_cnn_model(input_shape, l2_reg=0.001):
    """
    CNN (Convolutional Neural Network)
    より深い構造で空間パターンを捉える
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # 畳み込みブロック1
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # 畳み込みブロック2
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # 全結合層
            layers.Flatten(),
            layers.Dense(
                128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


# ==================================================
# 🚀 メイン処理
# ==================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Transfer Learning: FCN vs CNN (q=2 → q=3)")
    print("=" * 70)

    # 1. データ読み込み
    print("\n📂 Loading Training Data (q=2, Ising)...")
    X_train_raw, y_train, train_betas = load_train_data(TRAIN_FOLDER_NAME)

    print("\n📂 Loading Test Data (q=3, Potts)...")
    X_test_raw, test_betas = load_test_data_with_beta(TEST_FOLDER_NAME)

    if X_train_raw is None or X_test_raw is None:
        exit(1)

    # クラス分布確認
    n_ordered = np.sum(y_train == 1)
    n_disordered = np.sum(y_train == 0)
    print(f"\n📊 Training data distribution:")
    print(f"   Ordered phase (1): {n_ordered} samples")
    print(f"   Disordered phase (0): {n_disordered} samples")

    # --- データ整形 ---
    # CNN用
    X_train_cnn = X_train_raw.reshape(-1, L, L, 1)
    X_test_cnn = X_test_raw.reshape(-1, L, L, 1)

    # FCN用
    X_train_fcn = X_train_raw.reshape(-1, L * L)
    X_test_fcn = X_test_raw.reshape(-1, L * L)

    # FCNのみ標準化
    scaler = StandardScaler()
    X_train_fcn = scaler.fit_transform(X_train_fcn)
    X_test_fcn = scaler.transform(X_test_fcn)

    # メトリクス保存用
    metrics_data = []
    
    # q=3 Pottsの臨界温度
    tc_q3 = 1.0 / np.log(1 + np.sqrt(3))
    beta_c_q3 = 1.0 / tc_q3

    # ==========================================
    # 🧠 モデル1: FCN
    # ==========================================
    print("\n" + "=" * 70)
    print("🧠 [1/2] Training FCN Model")
    print("=" * 70)

    fcn_model = build_fcn_model(L * L, hidden_units=100, l2_reg=0.001)
    fcn_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    fcn_model.summary()

    # Callbacks
    early_stop_fcn = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    epoch_timer_fcn = EpochTimeCallback()

    history_fcn = fcn_model.fit(
        X_train_fcn,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop_fcn, epoch_timer_fcn],
        verbose=1,
    )

    # FCNメトリクス計算
    fcn_avg_epoch_time = epoch_timer_fcn.get_average_epoch_time()
    fcn_total_time = epoch_timer_fcn.get_total_time()
    fcn_test_acc = calculate_test_accuracy(fcn_model, X_test_fcn, test_betas, beta_c_q3)
    
    metrics_data.append(['FCN', fcn_avg_epoch_time, fcn_total_time, fcn_test_acc])
    
    print(f"\n📊 FCN Training Summary:")
    print(f"   Average epoch time: {fcn_avg_epoch_time:.2f}s")
    print(f"   Total training time: {fcn_total_time:.2f}s")
    print(f"   Test accuracy (q=3): {fcn_test_acc:.4f}")

    # ==========================================
    # 🧠 モデル2: CNN
    # ==========================================
    print("\n" + "=" * 70)
    print("🧠 [2/2] Training CNN Model")
    print("=" * 70)

    cnn_model = build_cnn_model((L, L, 1), l2_reg=0.001)
    cnn_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    cnn_model.summary()

    # Callbacks
    early_stop_cnn = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    epoch_timer_cnn = EpochTimeCallback()

    history_cnn = cnn_model.fit(
        X_train_cnn,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop_cnn, epoch_timer_cnn],
        verbose=1,
    )

    # CNNメトリクス計算
    cnn_avg_epoch_time = epoch_timer_cnn.get_average_epoch_time()
    cnn_total_time = epoch_timer_cnn.get_total_time()
    cnn_test_acc = calculate_test_accuracy(cnn_model, X_test_cnn, test_betas, beta_c_q3)
    
    metrics_data.append(['CNN', cnn_avg_epoch_time, cnn_total_time, cnn_test_acc])
    
    print(f"\n📊 CNN Training Summary:")
    print(f"   Average epoch time: {cnn_avg_epoch_time:.2f}s")
    print(f"   Total training time: {cnn_total_time:.2f}s")
    print(f"   Test accuracy (q=3): {cnn_test_acc:.4f}")

    # ==========================================
    # 💾 メトリクスをファイルに保存
    # ==========================================
    print("\n" + "=" * 70)
    print("💾 Saving Metrics to Files")
    print("=" * 70)
    
    save_metrics_to_csv(metrics_data, "training_metrics.csv")
    save_metrics_to_txt(metrics_data, "training_metrics.txt")

    # ==========================================
    # 📊 比較グラフ作成
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 Creating Comparison Graphs")
    print("=" * 70)
    
    def extract_simplified_config(folder_path):
        dir_name = os.path.basename(os.path.dirname(folder_path))
        match = re.search(r"(?P<bc>PBC|OBC)_(?P<map>.+)_(?P<std>Std|NoStd)", dir_name)
        if match:
            return f"{match.group('map')}_{match.group('bc')}_{match.group('std')}"
        return "Unknown_Config"
    
    simple_config_name = extract_simplified_config(TRAIN_FOLDER_NAME)
    plot_comparison_graphs(metrics_data, simple_config_name)

    # ==========================================
    # 🔮 推論と集計 (既存のグラフ)
    # ==========================================
    print("\n🔮 Predicting on q=3 (Potts) test data...")

    pred_fcn = fcn_model.predict(X_test_fcn, verbose=0).flatten()
    pred_cnn = cnn_model.predict(X_test_cnn, verbose=0).flatten()

    # 温度ごとの平均
    results = {}
    for i, beta in enumerate(test_betas):
        if beta not in results:
            results[beta] = {"fcn": [], "cnn": []}
        results[beta]["fcn"].append(pred_fcn[i])
        results[beta]["cnn"].append(pred_cnn[i])

    sorted_betas = sorted(results.keys())
    temps = [1.0 / b for b in sorted_betas]
    avg_fcn = [np.mean(results[b]["fcn"]) for b in sorted_betas]
    avg_cnn = [np.mean(results[b]["cnn"]) for b in sorted_betas]
    std_fcn = [np.std(results[b]["fcn"]) for b in sorted_betas]
    std_cnn = [np.std(results[b]["cnn"]) for b in sorted_betas]

    # ==========================================
    # 📊 既存のグラフ (Transfer Learning Result)
    # ==========================================
    save_filename = f"Transfer_Result_{simple_config_name}_critical.png"

    print(f"📌 Config Label: {simple_config_name}")
    print(f"💾 Output filename: {save_filename}")

    # データ準備
    prob_low_fcn = np.array(avg_fcn)
    prob_low_cnn = np.array(avg_cnn)
    prob_high_fcn = 1.0 - prob_low_fcn
    prob_high_cnn = 1.0 - prob_low_cnn

    # 描画範囲
    T_MIN, T_MAX = 0.94, 1.13
    tc_q2 = 1.0 / np.log(1 + np.sqrt(2))  # Ising (q=2) ≈ 1.13

    # プロット作成 (16:9 ワイド設定)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

    fig.suptitle(
        f"Transfer Learning Result (q=2 $\\to$ q=3)\n: {simple_config_name}",
        fontsize=24,
        fontweight="bold",
        y=0.92,
    )

    # 共通の描画関数
    def plot_prob_curves(ax, temps, p_low, p_high, std_low, sub_title):
        # 無秩序相 (High T)
        ax.plot(
            temps,
            p_high,
            "o-",
            color="red",
            label="Disordered (High T)",
            markersize=6,
            linewidth=2.5,
            alpha=0.8,
        )

        # 秩序相 (Low T)
        ax.plot(
            temps,
            p_low,
            "o-",
            color="blue",
            label="Ordered (Low T)",
            markersize=6,
            linewidth=2.5,
            alpha=0.8,
        )

        # エラーバー
        ax.errorbar(temps, p_low, yerr=std_low, fmt="none", ecolor="blue", alpha=0.3)

        # 臨界温度ライン
        ax.axvline(
            x=tc_q2,
            color="gray",
            linestyle="--",
            linewidth=2.5,
            alpha=0.6,
            label=f"Train: Ising $T_c$ ({tc_q2:.2f})",
        )
        ax.axvline(
            x=tc_q3,
            color="green",
            linestyle="-",
            linewidth=2.5,
            alpha=0.8,
            label=f"Target: Potts $T_c$ ({tc_q3:.2f})",
        )

        # デザイン調整
        ax.set_title(sub_title, fontsize=20, fontweight="bold", pad=15)
        ax.set_xlabel("Temperature $T$", fontsize=18)
        ax.set_xlim(T_MIN, T_MAX)
        ax.set_ylim(-0.05, 1.05)

        # 目盛り調整
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xticks(np.arange(T_MIN, T_MAX + 0.1, 0.1))
        ax.grid(True, linestyle=":", alpha=0.6)

    # 左: FCN
    plot_prob_curves(axes[0], temps, prob_low_fcn, prob_high_fcn, std_fcn, "FCN Model")
    axes[0].set_ylabel("Probability", fontsize=18)

    # 右: CNN
    plot_prob_curves(axes[1], temps, prob_low_cnn, prob_high_cnn, std_cnn, "CNN Model")

    # 凡例 & 保存
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        fontsize=16,
        frameon=True,
        edgecolor="black",
    )

    plt.subplots_adjust(top=0.80, bottom=0.18, wspace=0.15, left=0.08, right=0.95)

    plt.savefig(save_filename, dpi=150)
    print(f"\n✅ Transfer Learning graph saved as: {save_filename}")
    plt.show()

    # ==========================================
    # 📈 結果サマリー
    # ==========================================
    print("\n" + "=" * 70)
    print("📈 Final Results Summary")
    print("=" * 70)

    print(f"\n{'Model':<10} {'Avg Epoch Time':<18} {'Total Time':<15} {'Test Accuracy':<15}")
    print("-" * 70)
    for row in metrics_data:
        print(f"{row[0]:<10} {row[1]:>12.2f}s       {row[2]:>10.2f}s      {row[3]:>10.4f}")
    
    print("\n" + "=" * 70)
    print("✅ All processing completed successfully!")
    print("=" * 70)