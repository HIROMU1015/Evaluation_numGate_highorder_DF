from __future__ import annotations

from typing import Any, Sequence, Tuple


def set_loglog_axes(
    ax: Any,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    *,
    xlabel_kwargs: dict[str, Any] | None = None,
    ylabel_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
) -> None:
    """log-log 軸とラベル/タイトルを設定する。"""
    # スケール設定
    ax.set_xscale("log")
    ax.set_yscale("log")
    if title is not None:
        # タイトル設定
        ax.set_title(title, **(title_kwargs or {}))
    if xlabel is not None:
        # X ラベル設定
        ax.set_xlabel(xlabel, **(xlabel_kwargs or {}))
    if ylabel is not None:
        # Y ラベル設定
        ax.set_ylabel(ylabel, **(ylabel_kwargs or {}))


def unique_legend_entries(
    handles: Sequence[Any],
    labels: Sequence[str],
) -> Tuple[list[Any], list[str]]:
    """凡例の重複を除き、順序を維持して返す。"""
    # 表示済みラベルを管理
    seen = set()
    handles_u = []
    labels_u = []
    for h, lab in zip(handles, labels):
        if lab and lab not in seen:
            handles_u.append(h)
            labels_u.append(lab)
            seen.add(lab)
    return handles_u, labels_u
