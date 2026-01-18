from __future__ import annotations

import os
import pickle
from typing import Any, Optional

from config import PICKLE_DIR, PICKLE_DIR_GROUPED


def save_data(file_name: str, data: Any, gr: Optional[bool] = None):
    """pickle で結果を保存する（保存先は定数で一元化）。"""
    current_dir = os.getcwd()
    parent_dir = os.path.join(
        current_dir, PICKLE_DIR if gr is None else PICKLE_DIR_GROUPED
    )
    os.makedirs(parent_dir, exist_ok=True)
    file_path = os.path.join(parent_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"データを {file_path} に保存しました。")


def load_data(file_name): # フィッティング結果読み込み用
    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, PICKLE_DIR_GROUPED)
    file_path = os.path.join(parent_dir, file_name)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def label_replace(labelkey):  # 凡例用
    replacedir = {
        "4th(new_2)": "4th (new_2)",
        "4th(new_3)": "4th (new_3)",
        "8th(Morales)": "8th (Morales et al.)",
        "10th(Morales)": "10th (Morales et al.)",
        "8th(Yoshida)": "8th (Yoshida)",
    }
    if labelkey in replacedir.keys():
        labelkey = replacedir[labelkey]
    return labelkey
