from __future__ import annotations

import os
import tempfile
from multiprocessing import Pool
from typing import List, Optional

import scipy.sparse as sp


def sparse_matrix_multiply_from_folder(
    core_num: int, folder_path: str, file_names: List[str]
) -> Optional[str]:
    """フォルダ内の疎行列を順次乗算し、一時 npz に保存してパスを返す。"""
    if not file_names:
        return None
    # 先頭から順に乗算
    result = sp.load_npz(os.path.join(folder_path, file_names[0]))
    for name in file_names[1:]:
        result = result @ sp.load_npz(os.path.join(folder_path, name))
    # 結果を一時ファイルに保存
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_core_{core_num}.npz")
    sp.save_npz(tmp.name, result)
    return tmp.name


def multi_parallel_sparse_matrix_multiply_recursive(
    initial_folder_path: str, file_names: List[str], num_partitions: int
) -> sp.spmatrix:
    """疎行列の多段並列乗算。分割→各分割を乗算→一時ファイルへ。最後に連鎖乗算して返す。"""
    current_folder_path = initial_folder_path
    step = 1
    while len(file_names) >= 2:
        print(f"{step}回目の処理開始")
        # ファイルを分割して並列処理
        partition_size = (len(file_names) + num_partitions - 1) // num_partitions
        partitions = [
            file_names[i * partition_size : (i + 1) * partition_size]
            for i in range(num_partitions)
        ]
        with Pool(processes=num_partitions) as pool:
            results = pool.starmap(
                sparse_matrix_multiply_from_folder,
                [
                    (core, current_folder_path, part)
                    for core, part in enumerate(partitions)
                ],
            )
        # 有効結果だけに絞り、core番号順に並べ替え
        file_names = sorted(
            [p for p in results if p is not None],
            key=lambda p: int(p.split("_core_")[-1].split(".")[0]),
        )
        # 次ステップは一時ファイル群を入力にする
        current_folder_path = tempfile.gettempdir()  # 以降は一時ファイル群を入力にする
        num_partitions = max(1, num_partitions // 2)
        step += 1

    # 最終段：残ったファイルを逐次乗算して返す
    final_result = sp.load_npz(file_names[0])
    print("final calculating")
    for fp in file_names[1:]:
        final_result = final_result @ sp.load_npz(fp)
    print("done")
    return final_result
