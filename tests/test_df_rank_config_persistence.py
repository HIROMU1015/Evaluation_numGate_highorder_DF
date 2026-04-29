from __future__ import annotations

from pathlib import Path

from trotterlib import config


def test_persist_df_rank_selection_config_updates_target_file(tmp_path: Path) -> None:
    temp_config = tmp_path / "config.py"
    temp_config.write_text(
        "from typing import Dict\n"
        "DF_RANK_SELECTION_BY_MOLECULE: Dict[int, Dict[str, float | int | str]] = {2: {'rank_fraction': 0.5, 'selected_rank': 2, 'full_rank': 4, 'rank_ratio': '2/4'}}\n"
        "DF_RANK_SELECTION_BY_MOLECULE_x10 : Dict[int, Dict[str, float | int | str]] = {3: {'rank_fraction': 0.1, 'selected_rank': 1, 'full_rank': 9, 'rank_ratio': '1/9'}}\n",
        encoding="utf-8",
    )

    original_selection = dict(config.DF_RANK_SELECTION_BY_MOLECULE)
    original_fraction = dict(config.DF_RANK_FRACTION_BY_MOLECULE)
    try:
        config.DF_RANK_SELECTION_BY_MOLECULE.clear()
        config.DF_RANK_SELECTION_BY_MOLECULE.update(
            {
                13: {
                    "rank_fraction": 0.14792899408284024,
                    "selected_rank": 25,
                    "full_rank": 169,
                    "rank_ratio": "25/169",
                    "target_error_ha": 1.5936001019904e-4,
                }
            }
        )
        config.DF_RANK_FRACTION_BY_MOLECULE.clear()
        config.DF_RANK_FRACTION_BY_MOLECULE[13] = 0.14792899408284024

        written_path = config.persist_df_rank_selection_config(temp_config)
        assert written_path == temp_config.resolve()
        rewritten = temp_config.read_text(encoding="utf-8")
        assert "13:" in rewritten
        assert "'rank_fraction': 0.14792899408284024" in rewritten
        assert "'target_error_ha': 0.00015936001019904" in rewritten
        assert "DF_RANK_SELECTION_BY_MOLECULE_x10" in rewritten
    finally:
        config.DF_RANK_SELECTION_BY_MOLECULE.clear()
        config.DF_RANK_SELECTION_BY_MOLECULE.update(original_selection)
        config.DF_RANK_FRACTION_BY_MOLECULE.clear()
        config.DF_RANK_FRACTION_BY_MOLECULE.update(original_fraction)
