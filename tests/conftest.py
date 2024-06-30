from typing import Any, Dict

import pyrootutils
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
import sys
from src.utils import register_custom_resolvers
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
}


@pytest.fixture(scope="package")
@register_custom_resolvers(config_name="train_baseline.yaml", **_HYDRA_PARAMS)
def cfg_baseline_global() -> DictConfig:
    with initialize_config_dir(
        version_base=_HYDRA_PARAMS["version_base"],
        config_dir=_HYDRA_PARAMS["config_path"],
    ):
        cfg = compose(
            config_name="train_baseline.yaml",
            return_hydra_config=True,
            overrides=[],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_baseline(cfg_baseline_global, tmp_path) -> DictConfig:
    cfg = cfg_baseline_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()



@pytest.fixture(scope="package")
def progress_bar_status_message() -> Dict[str, Any]:
    return {
        "Info": "5/10 [######9     ] 59%",
        "Time/D": 0.9999999999999999999999,
        "Time/B": 0.1238476123864126345187245,
        "Time/AvgD": 0.812341348176234987162349817643,
        "Time/AvgB": 0.8561283745187634581726345,
        "Loss/L1": 112398746192834619827364,
        "Loss/L2": 19234618623.0,
        "Loss/L2.5": 1,
        "Loss/L3": 0.8126347152,
        "Loss/L4": 10,
        "Loss/L5": float("inf"),
        "Metrics/One": 0.888888,
        "Metrics/Two": 123987.0,
        "Metrics/Three": 1.0,
    }
