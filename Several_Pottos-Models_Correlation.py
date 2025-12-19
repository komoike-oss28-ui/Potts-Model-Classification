import numpy as np
import os
import glob
import time

# ==================================================
# ⚙️ 設定セクション
# ==================================================
Q_PARAM = 3
TARGET_R = 32
MAP_TYPES_TO_RUN = ["fix_origin", "spatial_avg", "spatial_map"]
BC_SETTINGS = [True]
INPUT_FOLDER = f"potts_q{Q_PARAM}_critical_data_opt"
OUTPUT_PREFIX = f"maps_POTTS_q{Q_PARAM}_R{TARGET_R}_critical"

# ベクトル生成設定
USE_RAW_PBC_DIST = True
SKIP_EVEN = False
MAX_R_OBC = 45
L = 64
ORIGIN = None


# ==================================================
def get_unique_vectors(limit_range, L, use_pbc, skip_even):
    """提示されたベクトル生成関数"""
    groups = {}

    if use_pbc and USE_RAW_PBC_DIST:
        search_lim = L - 1
    elif use_pbc:
        search_lim = L // 2
    else:
        search_lim = int(limit_range)

    search_range = range(-search_lim, search_lim + 1)
    temp_groups = {}

    for x in search_range:
        for y in search_range:
            if x == 0 and y == 0:
                continue

            if use_pbc and USE_RAW_PBC_DIST:
                raw_dist = np.sqrt(x**2 + y**2)
                if raw_dist > limit_range + 0.0001:
                    continue
                dist_val = raw_dist
            elif use_pbc:
                rx_eff = (x + L // 2) % L - L // 2
                ry_eff = (y + L // 2) % L - L // 2
                dist_val = np.sqrt(rx_eff**2 + ry_eff**2)
            else:
                dist_val = np.sqrt(x**2 + y**2)

            if dist_val > limit_range:
                continue

            if skip_even:
                is_small = dist_val < 2.99
                if not is_small and int(dist_val) % 2 == 0:
                    continue

            if dist_val < 2.99:
                key = f"{dist_val:.4f}"
            else:
                key = f"{int(dist_val):.4f}"

            if key not in temp_groups:
                temp_groups[key] = []

            if use_pbc:
                rx_shift = (x + L // 2) % L - L // 2
                ry_shift = (y + L // 2) % L - L // 2
            else:
                rx_shift, ry_shift = x, y

            eff_dist = np.sqrt(rx_shift**2 + ry_shift**2)
            temp_groups[key].append({"vec": (rx_shift, ry_shift), "eff": eff_dist})

    for key, candidates in temp_groups.items():
        if not candidates:
            continue
        min_eff = min(c["eff"] for c in candidates)
        clean_vectors = []
        seen_vecs = set()
        for c in candidates:
            if abs(c["eff"] - min_eff) < 0.99999:
                v = c["vec"]
                if v not in seen_vecs:
                    clean_vectors.append(v)
                    seen_vecs.add(v)
        groups[key] = clean_vectors

    return groups


def calculate_potts_correlation_ultra_opt(
    spins, r_vec_list, q, use_pbc, map_type, origin
):
    """
    超最適化された相関計算
    - fix_originを完全ベクトル化
    - 不要な型変換を削減
    - メモリアクセスパターンを最適化
    """
    L_size = spins.shape[0]
    scale_factor = 1.0 / (q - 1)

    # --- Paper Style (roll利用、最適化済み) ---
    if map_type == "spatial_map":
        accumulated_map = np.zeros((L_size, L_size), dtype=np.float32)

        for rx, ry in r_vec_list:
            shifted = np.roll(spins, (-rx, -ry), axis=(0, 1))
            # 直接float32で計算（bool→int変換を省略）
            accumulated_map += (q * (spins == shifted) - 1) * scale_factor

        return accumulated_map / len(r_vec_list)

    # --- Spatial Avg (最適化済み) ---
    elif map_type == "spatial_avg":
        corrs = np.empty(len(r_vec_list), dtype=np.float32)

        for idx, (rx, ry) in enumerate(r_vec_list):
            if use_pbc:
                shifted = np.roll(spins, (-rx, -ry), axis=(0, 1))
                corrs[idx] = np.mean((q * (spins == shifted) - 1) * scale_factor)
            else:
                # OBC: スライス計算
                x0_s = max(0, -rx)
                x0_e = min(L_size, L_size - rx)
                y0_s = max(0, -ry)
                y0_e = min(L_size, L_size - ry)
                x1_s = max(0, rx)
                x1_e = min(L_size, L_size + rx)
                y1_s = max(0, ry)
                y1_e = min(L_size, L_size + ry)

                if x0_s < x0_e and y0_s < y0_e:
                    matches = spins[x0_s:x0_e, y0_s:y0_e] == spins[x1_s:x1_e, y1_s:y1_e]
                    corrs[idx] = np.mean((q * matches - 1) * scale_factor)
                else:
                    corrs[idx] = 0.0

        final_val = np.mean(corrs)
        return np.full((L_size, L_size), final_val, dtype=np.float32)

    # --- Fix Origin (★完全ベクトル化版★) ---
    elif map_type == "fix_origin":
        corr_map = np.zeros((L_size, L_size), dtype=np.float32)
        x0, y0 = origin if origin else (L_size // 2, L_size // 2)
        s0 = spins[x0, y0]

        if use_pbc:
            # PBC: 全ベクトルを一括処理
            r_vecs = np.array(r_vec_list, dtype=np.int32)
            tx = (x0 + r_vecs[:, 0]) % L_size
            ty = (y0 + r_vecs[:, 1]) % L_size

            # 一括で比較・計算
            is_same = s0 == spins[tx, ty]
            values = (q * is_same - 1) * scale_factor

            # 結果を格納
            corr_map[tx, ty] = values

        else:
            # OBC: 境界チェック付き
            r_vecs = np.array(r_vec_list, dtype=np.int32)
            tx = x0 + r_vecs[:, 0]
            ty = y0 + r_vecs[:, 1]

            # 境界内のインデックスをフィルタ
            valid_mask = (tx >= 0) & (tx < L_size) & (ty >= 0) & (ty < L_size)
            tx_valid = tx[valid_mask]
            ty_valid = ty[valid_mask]

            if len(tx_valid) > 0:
                is_same = s0 == spins[tx_valid, ty_valid]
                values = (q * is_same - 1) * scale_factor
                corr_map[tx_valid, ty_valid] = values

        return corr_map


# ==================================================
# メイン処理
# ==================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Ultra-Optimized Correlation Map Generation")
    print("=" * 70)

    input_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.npy")))

    if not input_files:
        print(f"❌ Error: No .npy files found in {INPUT_FOLDER}")
        exit(1)

    print(f"📂 Input folder: {INPUT_FOLDER}")
    print(f"📊 Total files: {len(input_files)}")
    print(f"🎯 Target R: {TARGET_R}")
    print(f"⚙️  Q parameter: {Q_PARAM}")
    print("=" * 70)

    total_start = time.time()

    for map_type in MAP_TYPES_TO_RUN:
        for is_pbc in BC_SETTINGS:
            bc_label = "PBC" if is_pbc else "OBC"

            # ベクトル取得
            limit_range = (L - 1) if (is_pbc and USE_RAW_PBC_DIST) else TARGET_R
            groups = get_unique_vectors(limit_range, L, is_pbc, SKIP_EVEN)

            target_key = f"{int(TARGET_R):.4f}"
            if target_key not in groups:
                print(f"⚠️  Warning: R={TARGET_R} not found in vector groups. Skipping.")
                continue

            vec_list = groups[target_key]

            # 出力フォルダ
            config_name = f"{bc_label}_{map_type}_NoStd"
            output_dir = f"{OUTPUT_PREFIX}_{config_name}/r_{float(TARGET_R):08.4f}"
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n📦 Processing: {config_name}")
            print(f"   Vectors: {len(vec_list)} vectors at R={TARGET_R}")
            print(f"   Output: {output_dir}")

            start_time = time.time()

            for idx, f_path in enumerate(input_files):
                # 進捗表示
                if (idx + 1) % 500 == 0 or idx == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    print(f"   [{idx+1:4d}/{len(input_files)}] {rate:.1f} files/s")

                fname = os.path.basename(f_path)
                spins = np.load(f_path)  # int8で読み込まれる

                # 相関マップ計算
                cmap = calculate_potts_correlation_ultra_opt(
                    spins, vec_list, Q_PARAM, is_pbc, map_type, ORIGIN
                )

                # float16で保存（メモリ削減）
                output_path = os.path.join(output_dir, fname)
                np.save(output_path, cmap.astype(np.float16))

            elapsed = time.time() - start_time
            print(
                f"   ✓ Completed in {elapsed:.2f}s ({len(input_files)/elapsed:.1f} files/s)"
            )

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"🎉 All processing completed!")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📁 Output prefix: {OUTPUT_PREFIX}_*")
    print("=" * 70)
