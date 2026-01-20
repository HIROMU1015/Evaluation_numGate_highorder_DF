from __future__ import annotations

import os
import pickle
from typing import Any, Optional
from pathlib import Path

from .config import PICKLE_DIR, PICKLE_DIR_GROUPED, pickle_dir


def save_data(file_name: str, data: Any, gr: bool | None = None) -> None:
    """キャッシュ先にデータを保存する。"""
    parent = pickle_dir(gr)
    parent.mkdir(parents=True, exist_ok=True)
    # 相対パスはキャッシュ配下に解決
    path = Path(file_name)
    if path.parent == Path(".") and not path.is_absolute():
        path = parent / file_name
    with path.open("wb") as f:
        pickle.dump(data, f)

def load_data(
    name: str,
    gr: bool | None = True,
    *,
    use_original: bool = False,
) -> Any:
    """キャッシュからデータを読み込む。"""
    parent = pickle_dir(gr if gr is not None else None, use_original=use_original)
    # 相対パスはキャッシュ配下に解決
    path = Path(name)
    if path.parent == Path(".") and not path.is_absolute():
        path = parent / name
    with path.open("rb") as f:
        return pickle.load(f)


def label_replace(labelkey: str) -> str:  # 凡例用
    """凡例表示用のラベルに変換する。"""
    # 表示名の置換ルール
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
