from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable


# =========================
# Project / Output paths
# =========================

def _find_project_root(start: Path) -> Path:
    """pyproject.toml or .git を目印に上へ辿って project root を推定する。"""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # 見つからなければ config.py のあるディレクトリを root 扱い
    return start


# 環境変数で明示できるように（CI や別配置でも事故らない）
_env_root = os.environ.get("TROTTER_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).expanduser().resolve()
else:
    PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# 生成物の “フォルダ名” は今の互換のため残す（既存コードが参照してても壊れない）
PICKLE_DIR = "trotter_expo_coeff"          # :contentReference[oaicite:2]{index=2}
PICKLE_DIR_GROUPED = "trotter_expo_coeff_gr"  # :contentReference[oaicite:3]{index=3}

# 実際に使う Path（ここを保存/読込の基準にする）
PICKLE_DIR_PATH = ARTIFACTS_DIR / PICKLE_DIR
PICKLE_DIR_GROUPED_PATH = ARTIFACTS_DIR / PICKLE_DIR_GROUPED

# もし matrix/ calculation/ も同じ階層に寄せたいなら
MATRIX_DIR = ARTIFACTS_DIR / "matrix"
CALCULATION_DIR = ARTIFACTS_DIR / "calculation"

def ensure_artifact_dirs() -> None:
    """必要な出力ディレクトリを作る。"""
    for d in (ARTIFACTS_DIR, PICKLE_DIR_PATH, PICKLE_DIR_GROUPED_PATH, MATRIX_DIR, CALCULATION_DIR):
        d.mkdir(parents=True, exist_ok=True)

def pickle_dir(gr: bool | None = None) -> Path:
    """gr(None)=通常, gr(True/False)= grouped 側、という今の呼び出し癖に合わせた切替。"""
    return PICKLE_DIR_PATH if gr is None else PICKLE_DIR_GROUPED_PATH


# =========================
# Existing constants (keep)
# =========================

DEFAULT_BASIS = "sto-3g"   # :contentReference[oaicite:4]{index=4}
DEFAULT_DISTANCE = 1.0     # :contentReference[oaicite:5]{index=5}

POOL_PROCESSES = 32        # :contentReference[oaicite:6]{index=6}

# QPE のユニタリ作用回数β (M = β / ε)
BETA = 1.2                 # :contentReference[oaicite:7]{index=7}

# 化学的精度
CA = 1.59360010199040e-3   # :contentReference[oaicite:8]{index=8}


# =========================
# Plot style (as-is)
# =========================

COLOR_MAP = {
    "2nd": "g",
    "4th(new_3)": "r",
    "4th(new_1)": "lightcoral",
    "4th(new_2)": "b",
    "6th(new_4)": "darkgreen",
    "4th": "c",
    "8th(Morales)": "m",
    "10th(Morales)": "greenyellow",
    "8th(Yoshida)": "orange",
}  # :contentReference[oaicite:9]{index=9}

MARKER_MAP = {
    "2nd": "o",
    "4th(new_3)": "v",
    "4th(new_1)": "lightcoral",  # ※ここは marker じゃなく色が入ってるので要注意（後で直すと◎）
    "4th(new_2)": "<",
    "6th(new_4)": "darkgreen",   # ※同上
    "4th": "^",
    "8th(Morales)": "h",
    "10th(Morales)": "H",
    "8th(Yoshida)": ">",
}  # :contentReference[oaicite:10]{index=10}


# =========================
# PF order / tables (as-is)
# =========================

P_DIR = {
    "6th(new_4)": 6,
    "6th(new_3)": 6,
    "4th(new_1)": 4,
    "4th(new_2)": 4,
    "4th(new_3)": 4,
    "2nd": 2,
    "4th": 4,
    "8th(Morales)": 8,
    "10th(Morales)": 10,
    "8th(Yoshida)": 8,
}  # :contentReference[oaicite:11]{index=11}


# =========================
# DECOMPO_NUM, PF_RZ_LAYER
# =========================

# PF 1 ステップに含まれるパウリローテーションの個数
_DECOMPO_NUM_KEYS = [
    "8th(Yoshida)",
    "2nd",
    "4th",
    "8th(Morales)",
    "10th(Morales)",
    "4th(new_3)",
    "4th(new_2)",
    "4(new_1)",
    "6(new_3)",
]
_DECOMPO_NUM_VALUES = {
    "H2":  [220, 24, 52, 248, 472, 108, 80, 52, 108],
    "H3":  [1476, 118, 312, 1670, 3222, 700, 506, 312, 700],
    "H4":  [5436, 396, 1116, 6156, 11916, 2556, 1836, 1116, 2556],
    "H5":  [14200, 998, 2884, 16086, 31174, 6656, 4770, 2884, 6656],
    "H6":  [30648, 2116, 6192, 34724, 67332, 14344, 10268, 6192, 14344],
    "H7":  [58920, 4026, 11868, 66762, 129498, 27552, 19710, 11868, 27552],
    "H8":  [102556, 6964, 20620, 116212, 225460, 47932, 34276, 20620, 47932],
    "H9":  [170016, 11494, 34140, 192662, 373830, 79432, 56786, 34140, 79432],
    "H10": [261960, 17660, 52560, 296860, 576060, 122360, 87460, 52560, 122360],
    "H11": [385648, 25946, 77332, 437034, 848122, 180104, 128718, 77332, 180104],
    "H12": [550620, 36988, 110364, 623996, 1211004, 257116, 183740, 110364, 257116],
    "H13": [767016, 51462, 153684, 869238, 1687014, 358128, 255906, 153684, 358128],
    "H14": [1037656, 69556, 207856, 1175956, 2282356, 484456, 346156, 207856, 484456],
    "H15": [1385520, 92802, 277476, 1570194, 3047586, 646824, 462150, 277476, 646824],
}
DECOMPO_NUM = {
    h: dict(zip(_DECOMPO_NUM_KEYS, vals))
    for h, vals in _DECOMPO_NUM_VALUES.items()
} # :contentReference[oaicite:12]{index=12}

# PF 1 ステップに含まれる RZ のレイヤー数
# Inoue グルーピングにより A → n にそのまま変換した場合
_PF_RZ_LAYER_KEYS = ['2nd', '4th', '8th(Morales)', '10th(Morales)', '8th(Yoshida)', '4th(new_3)', '4th(new_2)']

_PF_RZ_LAYER_VALUES = {
    'H2':  [9, 19, 89, 169, 79, 39, 29],
    'H3':  [39, 101, 535, 1031, 473, 225, 163],
    'H4':  [99, 279, 1539, 2979, 1359, 639, 459],
    'H5':  [341, 999, 5605, 10869, 4947, 2315, 1657],
    'H6':  [568, 1674, 9416, 18264, 8310, 3886, 2780],
    'H7':  [1064, 3158, 17816, 34568, 15722, 7346, 5252],
    'H8':  [1220, 3618, 20404, 39588, 18006, 8414, 6016],
    'H9':  [2442, 7280, 41146, 79850, 36308, 16956, 12118],
    'H10': [3172, 9464, 53508, 103844, 47216, 22048, 15756],
    'H11': [4511, 13479, 76255, 147999, 67287, 31415, 22447],
    'H12': [4865, 14535, 82225, 159585, 72555, 33875, 24205],
    'H13': [7476, 22362, 126564, 245652, 111678, 52134, 37248],
    'H14': [8527, 25511, 144399, 280271, 127415, 59479, 42495],
    'H15': [11657, 34895, 197561, 383465, 174323, 81371, 58133],
}

PF_RZ_LAYER = {
    h: dict(zip(_PF_RZ_LAYER_KEYS, vals))
    for h, vals in _PF_RZ_LAYER_VALUES.items()
} # :contentReference[oaicite:13]{index=13}


# =========================
# Optional: consistency check
# =========================

def _keys(*dicts: Dict) -> set:
    s: set = set()
    for d in dicts:
        s |= set(d.keys())
    return s

def validate_config() -> None:
    """
    PF 名の表記ゆれ（例: "4(new_1)" vs "4th(new_1)"）みたいな事故を早期に検出する。
    必要に応じて assert を増やしてOK。
    """
    # 例: P_DIR と style の整合
    missing_color = set(P_DIR) - set(COLOR_MAP)
    missing_marker = set(P_DIR) - set(MARKER_MAP)
    if missing_color or missing_marker:
        raise ValueError(f"Missing style keys: color={missing_color}, marker={missing_marker}")
