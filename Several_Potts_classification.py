import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==================================================
# ⚙️ 設定: フォルダパスとパラメータ
# ==================================================
# 学習用: q=2 (Ising)
TRAIN_FOLDER_NAME = 'maps_POTTS_q2_R32_OBC_fix_origin_NoStd/r_32.0000'
# テスト用: q=3 (Potts)
TEST_FOLDER_NAME  = 'maps_POTTS_q3_R32_OBC_fix_origin_NoStd/r_32.0000'

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
# 🛠️ ヘルパー関数
# ==================================================
def extract_beta(filename):
    """
    ファイル名からbeta値を抽出
    想定形式: conf_low_b0.1234_n001.npy または conf_high_b2.3456_n001.npy
    """
    # 'b' の後の数値を抽出
    match = re.search(r'_b([0-9.]+)_', filename)
    if match:
        return float(match.group(1))
    
    # フォールバック: beta_X.XXX 形式も試す
    match = re.search(r'beta_?([0-9.]+)', filename)
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
        
        data = np.load(f).astype('float32')
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
        
        data = np.load(f).astype('float32')
        X.append(data)
        betas.append(beta)
    
    print(f"   Loaded {len(X)} test samples")
    return np.array(X), np.array(betas)

# ==================================================
# 🧠 モデル定義
# ==================================================
def build_fcn_model(input_dim, hidden_units=100, l2_reg=0.001):
    """
    FCN (Fully Connected Network)
    論文ベースだが正則化を追加
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn_model(input_shape, l2_reg=0.001):
    """
    CNN (Convolutional Neural Network)
    より深い構造で空間パターンを捉える
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # 畳み込みブロック1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # 畳み込みブロック2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # 全結合層
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
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
    X_test_cnn  = X_test_raw.reshape(-1, L, L, 1)
    
    # FCN用
    X_train_fcn = X_train_raw.reshape(-1, L * L)
    X_test_fcn  = X_test_raw.reshape(-1, L * L)
    
    # FCNのみ標準化
    scaler = StandardScaler()
    X_train_fcn = scaler.fit_transform(X_train_fcn)
    X_test_fcn  = scaler.transform(X_test_fcn)
    
    # ==========================================
    # 🧠 モデル1: FCN
    # ==========================================
    print("\n" + "=" * 70)
    print("🧠 [1/2] Training FCN Model")
    print("=" * 70)
    
    fcn_model = build_fcn_model(L * L, hidden_units=100, l2_reg=0.001)
    fcn_model.compile(optimizer='adam', 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
    
    fcn_model.summary()
    
    # Early Stopping
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, 
                                         restore_best_weights=True)
    
    history_fcn = fcn_model.fit(
        X_train_fcn, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )
    
    # ==========================================
    # 🧠 モデル2: CNN
    # ==========================================
    print("\n" + "=" * 70)
    print("🧠 [2/2] Training CNN Model")
    print("=" * 70)
    
    cnn_model = build_cnn_model((L, L, 1), l2_reg=0.001)
    cnn_model.compile(optimizer='adam', 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
    
    cnn_model.summary()
    
    history_cnn = cnn_model.fit(
        X_train_cnn, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )
    
    # ==========================================
    # 🔮 推論と集計
    # ==========================================
    print("\n🔮 Predicting on q=3 (Potts) test data...")
    
    pred_fcn = fcn_model.predict(X_test_fcn, verbose=0).flatten()
    pred_cnn = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    
    # 温度ごとの平均
    results = {}
    for i, beta in enumerate(test_betas):
        if beta not in results:
            results[beta] = {'fcn': [], 'cnn': []}
        results[beta]['fcn'].append(pred_fcn[i])
        results[beta]['cnn'].append(pred_cnn[i])
    
    sorted_betas = sorted(results.keys())
    temps = [1.0 / b for b in sorted_betas]
    avg_fcn = [np.mean(results[b]['fcn']) for b in sorted_betas]
    avg_cnn = [np.mean(results[b]['cnn']) for b in sorted_betas]
    std_fcn = [np.std(results[b]['fcn']) for b in sorted_betas]
    std_cnn = [np.std(results[b]['cnn']) for b in sorted_betas]
    
    # ==========================================
    # 📊 プロット
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 左: 訓練履歴 ---
    ax = axes[0]
    ax.plot(history_fcn.history['loss'], label='FCN Train Loss', color='orange', alpha=0.7)
    ax.plot(history_fcn.history['val_loss'], label='FCN Val Loss', color='orange', linestyle='--')
    ax.plot(history_cnn.history['loss'], label='CNN Train Loss', color='blue', alpha=0.7)
    ax.plot(history_cnn.history['val_loss'], label='CNN Val Loss', color='blue', linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- 右: 転移学習結果 ---
    ax = axes[1]
    
    # 標準偏差も表示
    ax.errorbar(temps, avg_fcn, yerr=std_fcn, fmt='o--', 
               label='FCN', color='tab:orange', markersize=6, alpha=0.8, capsize=3)
    ax.errorbar(temps, avg_cnn, yerr=std_cnn, fmt='s-', 
               label='CNN', color='tab:blue', markersize=6, alpha=0.8, capsize=3)
    
    # 臨界温度ライン
    tc_q2 = T_C_ISING
    ax.axvline(x=tc_q2, color='gray', linestyle='--', linewidth=1.5, 
              label=f'Ising $T_c$ (q=2): {tc_q2:.2f}')
    
    tc_q3 = 1.0 / np.log(1 + np.sqrt(3))
    ax.axvline(x=tc_q3, color='red', linestyle='-', linewidth=2, 
              label=f'Potts $T_c$ (q=3): {tc_q3:.2f}')
    
    ax.set_xlabel('Temperature $T$', fontsize=14)
    ax.set_ylabel('Order Probability', fontsize=14)
    ax.set_title('Transfer Learning: q=2 Model → q=3 Data', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    save_path = "transfer_comparison_FCN_vs_CNN.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graph saved as: {save_path}")
    plt.show()
    
    # ==========================================
    # 📈 結果サマリー
    # ==========================================
    print("\n" + "=" * 70)
    print("📈 Results Summary")
    print("=" * 70)
    
    # 最終訓練精度
    fcn_final_acc = history_fcn.history['accuracy'][-1]
    cnn_final_acc = history_cnn.history['accuracy'][-1]
    print(f"FCN final training accuracy: {fcn_final_acc:.4f}")
    print(f"CNN final training accuracy: {cnn_final_acc:.4f}")
    
    # Tc付近の予測値
    idx_closest = np.argmin(np.abs(np.array(temps) - tc_q3))
    print(f"\nPredictions near Potts Tc ({tc_q3:.2f}):")
    print(f"   FCN: {avg_fcn[idx_closest]:.3f} ± {std_fcn[idx_closest]:.3f}")
    print(f"   CNN: {avg_cnn[idx_closest]:.3f} ± {std_cnn[idx_closest]:.3f}")
    print("=" * 70)