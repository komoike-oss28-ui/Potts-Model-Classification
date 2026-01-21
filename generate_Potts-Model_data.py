import numpy as np
import os
import time

# ==================================================
# ⚙️ 設定
# ==================================================
L = 64
J = 1.0
q = 3  # Pottsモデルの状態数
# beta_values = list(np.linspace(beta_c * 0.95, beta_c * 1.05, 15)) :critical

# list(np.linspace(beta_low, beta_c * 0.8, 4 or 7)) :高温相
# + list(np.linspace(beta_c * 0.9, beta_c * 1.1, 6)) テスト時使用
# list(np.linspace(beta_c * 1.2, beta_high, 4 or 7)) :低温相

equilibration_mcs_ml = 7000
num_configs_per_temp = 200

output_ml_folder = f"potts_q{q}_critical_data_opt"
os.makedirs(output_ml_folder, exist_ok=True)

# 臨界温度 T_c = 1 / ln(1 + sqrt(q))
val = 1.0 + np.sqrt(q)
T_c = 1.0 / np.log(val)
beta_c = 1.0 / T_c  # 0.44

# 温度設定 (臨界点周辺)
beta_low = beta_c * 0.4
beta_high = beta_c * 1.6
beta_values = list(np.linspace(beta_c * 0.95, beta_c * 1.05, 15))

print(f"Potts q={q}, Tc={T_c:.4f}, beta_c={beta_c:.4f}")


# ==================================================
# ⚡️ 高速化ロジック (チェッカーボード更新)
# ==================================================
def update_checkerboard_optimized(spins, beta, q, J, mask):
    """
    市松模様の片側を一括更新 (最適化版)
    コピーを避け、必要な計算のみ実行
    """
    L = spins.shape[0]
    N_sub = np.sum(mask)

    # マスク位置のインデックス取得
    indices = np.where(mask)
    i_coords = indices[0]
    j_coords = indices[1]

    # 現在のスピン値
    current_spins = spins[mask]

    # 4近傍の座標 (周期境界条件)
    neighbors = [
        ((i_coords - 1) % L, j_coords),  # up
        ((i_coords + 1) % L, j_coords),  # down
        (i_coords, (j_coords - 1) % L),  # left
        (i_coords, (j_coords + 1) % L),  # right
    ]

    # 現在の一致数
    current_matches = np.zeros(N_sub, dtype=np.int32)
    for ni, nj in neighbors:
        current_matches += (spins[ni, nj] == current_spins).astype(np.int32)

    # 新しいスピン提案
    new_spins = np.random.randint(0, q, size=N_sub)

    # 提案後の一致数
    new_matches = np.zeros(N_sub, dtype=np.int32)
    for ni, nj in neighbors:
        new_matches += (spins[ni, nj] == new_spins).astype(np.int32)

    # エネルギー差
    dE = -J * (new_matches - current_matches)

    # メトロポリス判定
    # dE <= 0 なら常に受理、dE > 0 なら確率 exp(-beta * dE) で受理
    accept_prob = np.exp(-beta * dE)
    accept_prob[dE <= 0] = 1.0

    r = np.random.rand(N_sub)
    accept_mask = r < accept_prob

    # 受理された更新のみ適用
    spins[i_coords[accept_mask], j_coords[accept_mask]] = new_spins[accept_mask]

    return spins


def generate_configs_fast(beta, q, num_configs, total_mcs, equilibration_mcs):
    T = 1.0 / beta
    spins = np.random.randint(0, q, size=(L, L))

    # チェッカーボードマスク作成
    x = np.arange(L)[:, None]
    y = np.arange(L)[None, :]
    checker_board = (x + y) % 2
    mask_white = checker_board == 0
    mask_black = checker_board == 1

    config_count = 0
    phase_label = "low" if beta > beta_c else "high"

    # 保存間隔を正確に計算
    production_mcs = total_mcs - equilibration_mcs
    save_interval = production_mcs // num_configs

    print(
        f"  beta={beta:.4f} T={T:.4f} ({phase_label}, save every {save_interval} MCS)...",
        end="",
        flush=True,
    )
    st = time.time()

    configs_saved = []

    for mcs in range(1, total_mcs + 1):
        # 1 MCS = 白更新 + 黒更新
        spins = update_checkerboard_optimized(spins, beta, q, J, mask_white)
        spins = update_checkerboard_optimized(spins, beta, q, J, mask_black)

        # 保存ロジック (平衡化後、等間隔で保存)
        if mcs > equilibration_mcs:
            steps_since_eq = mcs - equilibration_mcs
            if steps_since_eq % save_interval == 0 and config_count < num_configs:
                fname = os.path.join(
                    output_ml_folder,
                    f"conf_{phase_label}_b{beta:.4f}_n{config_count:03d}.npy",
                )
                np.save(fname, spins.astype(np.int8))
                configs_saved.append(mcs)
                config_count += 1

    elapsed = time.time() - st
    mcs_per_sec = total_mcs / elapsed
    print(
        f" Done: {config_count} configs saved ({elapsed:.2f}s, {mcs_per_sec:.1f} MCS/s)"
    )

    return configs_saved


# ==================================================
# メイン
# ==================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Potts Model (q={q}) Monte Carlo Simulation")
    print(f"{'='*60}")
    print(f"Lattice size: {L}x{L}")
    print(f"Total MCS: {total_mcs_ml}")
    print(f"Equilibration: {equilibration_mcs_ml} MCS")
    print(f"Configs per temperature: {num_configs_per_temp}")
    print(f"Output folder: {output_ml_folder}/")
    print(f"{'='*60}\n")

    all_times = []
    for idx, beta in enumerate(beta_values):
        print(f"[{idx+1}/{len(beta_values)}]", end=" ")
        st = time.time()
        generate_configs_fast(
            beta, q, num_configs_per_temp, total_mcs_ml, equilibration_mcs_ml
        )
        all_times.append(time.time() - st)

    print(f"\n{'='*60}")
    print(f"All simulations completed!")
    print(f"Average time per temperature: {np.mean(all_times):.2f}s")
    print(f"Total time: {np.sum(all_times):.2f}s")
    print(f"{'='*60}\n")
