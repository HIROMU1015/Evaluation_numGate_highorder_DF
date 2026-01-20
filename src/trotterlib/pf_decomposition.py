from __future__ import annotations

from typing import Iterable, Sequence


def _iter_s2_steps(num_terms: int, w: float) -> Iterable[tuple[int, float]]:
    """単一 w の S2 対称ステップを生成する。"""
    for i in range(num_terms - 1):
        yield i, w / 2
    yield num_terms - 1, w
    for k in reversed(range(0, num_terms - 1)):
        yield k, w / 2


def _iter_left_steps(
    num_terms: int, w_max: float, w_next: float
) -> Iterable[tuple[int, float]]:
    """複数 w の左端ステップを生成する。"""
    for i in range(num_terms - 1):
        yield i, w_max / 2
    yield num_terms - 1, w_max
    for k in reversed(range(1, num_terms - 1)):
        yield k, w_max / 2
    yield 0, (w_max + w_next) / 2


def _iter_middle_steps(
    num_terms: int, w_first: float, w_second: float
) -> Iterable[tuple[int, float]]:
    """隣接する w の中間ステップを生成する。"""
    for i in range(1, num_terms - 1):
        yield i, w_first / 2
    yield num_terms - 1, w_first
    for k in reversed(range(1, num_terms - 1)):
        yield k, w_first / 2
    yield 0, (w_first + w_second) / 2


def _iter_right_steps(num_terms: int, w_last: float) -> Iterable[tuple[int, float]]:
    """複数 w の右端ステップを生成する。"""
    for i in range(1, num_terms - 1):
        yield i, w_last / 2
    yield num_terms - 1, w_last
    for k in reversed(range(0, num_terms - 1)):
        yield k, w_last / 2


def iter_pf_steps(
    num_terms: int,
    w_list: Sequence[float],
) -> Iterable[tuple[int, float]]:
    """Yield (index, weight) steps for symmetric product-formula decomposition."""
    if num_terms <= 0 or not w_list:
        return

    m = len(w_list)
    if m == 1:
        # 単一 w の S2 ステップ
        yield from _iter_s2_steps(num_terms, w_list[0])
        return

    # 先頭側のステップ
    yield from _iter_left_steps(num_terms, w_list[m - 1], w_list[m - 2])
    for i in reversed(range(1, m - 1)):
        # 中央の折り返し（前半）
        yield from _iter_middle_steps(num_terms, w_list[i], w_list[i - 1])
    for i in range(0, m - 1):
        # 中央の折り返し（後半）
        yield from _iter_middle_steps(num_terms, w_list[i], w_list[i + 1])
    # 末尾側のステップ
    yield from _iter_right_steps(num_terms, w_list[m - 1])
