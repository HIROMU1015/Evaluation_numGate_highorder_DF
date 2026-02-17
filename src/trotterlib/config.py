from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, TypeAlias


# =========================
# Project / Output paths
# =========================

def _find_project_root(start: Path) -> Path:
    """pyproject.toml or .git を目印に上へ辿って project root を推定する。"""
    # 上位ディレクトリを探索
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
PICKLE_DIR_GROUPED_ORIGINAL = "trotter_expo_coeff_gr_original"
PICKLE_DIR_DF = "trotter_expo_coeff_df"

# 実際に使う Path（ここを保存/読込の基準にする）
PICKLE_DIR_PATH = ARTIFACTS_DIR / PICKLE_DIR
PICKLE_DIR_GROUPED_PATH = ARTIFACTS_DIR / PICKLE_DIR_GROUPED
PICKLE_DIR_GROUPED_ORIGINAL_PATH = ARTIFACTS_DIR / PICKLE_DIR_GROUPED_ORIGINAL
PICKLE_DIR_DF_PATH = ARTIFACTS_DIR / PICKLE_DIR_DF

# もし matrix/ calculation/ も同じ階層に寄せたいなら
MATRIX_DIR = ARTIFACTS_DIR / "matrix"
CALCULATION_DIR = ARTIFACTS_DIR / "calculation"

def ensure_artifact_dirs(*, include_pickle_dirs: bool = True) -> None:
    """必要な出力ディレクトリを作る。"""
    # 生成対象のディレクトリを列挙
    dirs = [ARTIFACTS_DIR, MATRIX_DIR, CALCULATION_DIR]
    if include_pickle_dirs:
        dirs.extend([PICKLE_DIR_PATH, PICKLE_DIR_GROUPED_PATH, PICKLE_DIR_DF_PATH])
    # 必要なら作成
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def pickle_dir(gr: bool | None = None, *, use_original: bool = False) -> Path:
    """係数データの保存/読込先を返す（grouped/original を切替可能）。"""
    # gr=None は通常系、gr=True/False は grouped 系
    if gr is None:
        return PICKLE_DIR_PATH
    if use_original:
        return PICKLE_DIR_GROUPED_ORIGINAL_PATH
    return PICKLE_DIR_GROUPED_PATH


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

# 外挿などで使う target error
TARGET_ERROR = CA / 10

# DF の rank_fraction 推定に使う CCSD 誤差目標（化学的精度の 1/100）
DEFAULT_DF_CCSD_TARGET_ERROR_HA = CA / 100

# molecule_type -> DF rank_fraction (CCSD 誤差基準)。
# `trotterlib.ccsd.populate_df_rank_fraction_config(...)` で更新可能。
DF_RANK_FRACTION_BY_MOLECULE: Dict[int, float] = {
    2: 0.75, 
    3: 0.5555555555555556, 
    4: 0.4375, 
    5: 0.36, 
    6: 0.3055555555555556,
    7: 0.2653061224489796,
    }

# molecule_type -> DF rank selection metadata.
# 形式: {"rank_fraction": float, "selected_rank": int, "full_rank": int, "rank_ratio": "L/full"}
DF_RANK_SELECTION_BY_MOLECULE: Dict[int, Dict[str, float | int | str]] = {2: {'rank_fraction': 0.75, 'selected_rank': 3, 'full_rank': 4, 'rank_ratio': '3/4'}, 3: {'rank_fraction': 0.5555555555555556, 'selected_rank': 5, 'full_rank': 9, 'rank_ratio': '5/9'}, 4: {'rank_fraction': 0.4375, 'selected_rank': 7, 'full_rank': 16, 'rank_ratio': '7/16'}, 5: {'rank_fraction': 0.36, 'selected_rank': 9, 'full_rank': 25, 'rank_ratio': '9/25'}, 6: {'rank_fraction': 0.3055555555555556, 'selected_rank': 11, 'full_rank': 36, 'rank_ratio': '11/36'}, 7: {'rank_fraction': 0.2653061224489796, 'selected_rank': 13, 'full_rank': 49, 'rank_ratio': '13/49'}, 8: {'rank_fraction': 0.234375, 'selected_rank': 15, 'full_rank': 64, 'rank_ratio': '15/64'}, 9: {'rank_fraction': 0.20987654320987653, 'selected_rank': 17, 'full_rank': 81, 'rank_ratio': '17/81'}, 10: {'rank_fraction': 0.25, 'selected_rank': 25, 'full_rank': 100, 'rank_ratio': '25/100'}}


def get_df_rank_fraction_for_molecule(molecule_type: int) -> float | None:
    """設定済みの molecule_type 用 rank_fraction を返す。未設定なら None。"""
    return DF_RANK_FRACTION_BY_MOLECULE.get(int(molecule_type))


def get_df_rank_selection_for_molecule(
    molecule_type: int,
) -> Dict[str, float | int | str] | None:
    """設定済みの molecule_type 用 rank 選択情報を返す。未設定なら None。"""
    return DF_RANK_SELECTION_BY_MOLECULE.get(int(molecule_type))


def set_df_rank_fraction_for_molecule(
    molecule_type: int,
    rank_fraction: float,
    *,
    selected_rank: int | None = None,
    full_rank: int | None = None,
) -> None:
    """molecule_type 用 rank_fraction 設定を更新し、rank 選択情報も保持する。"""
    molecule_key = int(molecule_type)
    fraction_value = float(rank_fraction)
    DF_RANK_FRACTION_BY_MOLECULE[molecule_key] = fraction_value

    if full_rank is None:
        full_rank_value = max(1, molecule_key**2)
    else:
        full_rank_value = max(1, int(full_rank))
    if selected_rank is None:
        selected_rank_value = int(round(full_rank_value * fraction_value))
    else:
        selected_rank_value = int(selected_rank)
    selected_rank_value = max(1, min(selected_rank_value, full_rank_value))

    DF_RANK_SELECTION_BY_MOLECULE[molecule_key] = {
        "rank_fraction": fraction_value,
        "selected_rank": selected_rank_value,
        "full_rank": full_rank_value,
        "rank_ratio": f"{selected_rank_value}/{full_rank_value}",
    }


for _molecule_type, _fraction in list(DF_RANK_FRACTION_BY_MOLECULE.items()):
    set_df_rank_fraction_for_molecule(_molecule_type, _fraction)


# =========================
# Plot style
# =========================

@dataclass(frozen=True)
class PlotStyle:
    color: str
    marker: str


PLOT_STYLE: Dict[str, PlotStyle] = {
    "2nd": PlotStyle(color="g", marker="o"),
    "4th(new_3)": PlotStyle(color="r", marker="v"),
    "4th(new_1)": PlotStyle(color="lightcoral", marker="x"),
    "4th(new_2)": PlotStyle(color="b", marker="<"),
    "6th(new_4)": PlotStyle(color="darkgreen", marker="*"),
    "6th(new_3)": PlotStyle(color="seagreen", marker="s"),
    "4th": PlotStyle(color="c", marker="^"),
    "8th(Morales)": PlotStyle(color="m", marker="h"),
    "10th(Morales)": PlotStyle(color="greenyellow", marker="H"),
    "8th(Yoshida)": PlotStyle(color="orange", marker=">"),
}  # :contentReference[oaicite:9]{index=9}

COLOR_MAP = {key: style.color for key, style in PLOT_STYLE.items()}
MARKER_MAP = {key: style.marker for key, style in PLOT_STYLE.items()}


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

PFLabel: TypeAlias = str


_PF_LABEL_ALIASES = {
    "2nd": "2nd",
    "second": "2nd",
    "4th": "4th",
    "fourth": "4th",
    "4th(new_2)": "4th(new_2)",
    "4th(new_3)": "4th(new_3)",
    "8th": "8th(Morales)",
    "8th(morales)": "8th(Morales)",
    "8th_morales": "8th(Morales)",
    "8th-morales": "8th(Morales)",
    "8th (morales)": "8th(Morales)",
    "8th(yoshida)": "8th(Yoshida)",
    "8th_yoshida": "8th(Yoshida)",
    "8th-yoshida": "8th(Yoshida)",
    "8th (yoshida)": "8th(Yoshida)",
    "10th": "10th(Morales)",
    "10th(morales)": "10th(Morales)",
    "10th_morales": "10th(Morales)",
    "10th-morales": "10th(Morales)",
    "10th (morales)": "10th(Morales)",
}


def normalize_pf_label(num_w: PFLabel | None) -> PFLabel:
    """PF ラベル表記揺れを正規化し、正しければ返す。"""
    if num_w is None:
        raise KeyError(num_w)
    canonical = _PF_LABEL_ALIASES.get(str(num_w).strip().lower())
    if canonical is None:
        canonical = str(num_w)
    if canonical not in P_DIR:
        raise KeyError(num_w)
    return canonical


def require_pf_label(num_w: PFLabel | None) -> PFLabel:
    """PF ラベルを検証し、正しければ正規化して返す。"""
    return normalize_pf_label(num_w)


def pf_order(num_w: PFLabel | None) -> int:
    """PF ラベルから次数を返す。"""
    # P_DIR から次数を引く
    return P_DIR[require_pf_label(num_w)]


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
    """複数辞書のキー集合をまとめて返す。"""
    # キーを集合で合成
    s: set = set()
    for d in dicts:
        s |= set(d.keys())
    return s

def validate_config() -> None:
    """
    PF 名の表記ゆれ（例: "4(new_1)" vs "4th(new_1)"）みたいな事故を早期に検出する。
    必要に応じて assert を増やしてOK。
    """
    # スタイル定義のキー整合性を確認
    # 例: P_DIR と style の整合
    missing_color = set(P_DIR) - set(COLOR_MAP)
    missing_marker = set(P_DIR) - set(MARKER_MAP)
    if missing_color or missing_marker:
        raise ValueError(f"Missing style keys: color={missing_color}, marker={missing_marker}")
