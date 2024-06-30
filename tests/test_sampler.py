import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from hydra.utils import instantiate
from tests.helpers.run_if import RunIf
from typing import Any, Dict
import numpy as np 


def test_model_loading(cfg_baseline):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_baseline)
    with open_dict(cfg_baseline):
        sampler = instantiate(cfg_baseline.module.sampler)
        assert sampler is not None



def test_model_sampling(cfg_baseline):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_baseline)
    with open_dict(cfg_baseline):
        sampler = instantiate(cfg_baseline.module.sampler)
        seed = cfg_baseline.seed
        truncation = cfg_baseline.module.sampler.truncation
        components = cfg_baseline.module.sampler.components
        num = np.random.randint(0, components)
        start_layer = 0
        end_layer = sampler.model.get_max_latents()
        scale = 0.01
        samples = sampler.sample_latent(seed, truncation, num, scale, start_layer, end_layer)
        assert samples is not None
