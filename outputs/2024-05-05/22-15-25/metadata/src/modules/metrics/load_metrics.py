from typing import Tuple

import hydra
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection


def load_metrics(
    metrics_cfg: DictConfig,
) -> Tuple[Metric, Metric, MetricCollection]:
    """Load main metric, `best` metric tracker, MetricCollection of additional
    metrics.

    Args:
        metrics_cfg (DictConfig): Metrics config.

    Returns:
        Tuple[Metric, Metric, ModuleList]: Main metric, `best` metric tracker,
            MetricCollection of additional metrics.c
    """

    main_metric = hydra.utils.instantiate(metrics_cfg)


    return main_metric